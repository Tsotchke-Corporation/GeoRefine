"""One model-execution thread with bounded per-request delivery queues."""
from __future__ import annotations

from dataclasses import dataclass,field
import queue
import math
import threading


@dataclass
class Ticket:
    events: queue.Queue
    ready: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


class Service:
    def __init__(self,engine,*,device='metal',queue_events=128,wait_seconds=600,context_factory=None):
        if (type(queue_events) is not int or queue_events<1 or type(wait_seconds) not in (int,float)
                or not math.isfinite(wait_seconds) or wait_seconds<=0):
            raise ValueError('invalid service queue or timeout')
        self.engine=engine;self.device=device;self.queue_events=queue_events
        self.wait_seconds=wait_seconds;self.context_factory=context_factory
        self._commands=queue.Queue();self._tickets={};self._lock=threading.Lock()
        self._slots=threading.BoundedSemaphore(engine.max_requests)
        self._stopped=threading.Event()
        self._worker=threading.Thread(target=self._run,name='mlx-request-scheduler',daemon=True)
        self._worker.start()

    def _finish(self,request_id,ticket):
        with self._lock:
            if self._tickets.get(request_id) is not ticket:return False
            del self._tickets[request_id]
        self._slots.release()
        return True

    def submit(self,request_id,prompt,**kwargs):
        if not isinstance(request_id,str) or not request_id:
            raise ValueError('request ID must be a nonempty string')
        if not self._slots.acquire(blocking=False):raise MemoryError('request capacity exhausted')
        ticket=Ticket(queue.Queue(maxsize=self.queue_events))
        with self._lock:
            if self._stopped.is_set() or request_id in self._tickets:
                self._slots.release();raise ValueError('service closed or duplicate request ID')
            self._tickets[request_id]=ticket
            self._commands.put(('submit',request_id,ticket,prompt,kwargs))
        if not ticket.ready.wait(self.wait_seconds):
            self.cancel(request_id,ticket=ticket);raise TimeoutError('request admission timed out')
        if ticket.error is not None:raise ticket.error
        return ticket

    def cancel(self,request_id,*,ticket=None):
        with self._lock:
            current=self._tickets.get(request_id)
            if current is None or (ticket is not None and current is not ticket):return False
            current.cancelled.set()
            self._commands.put(('cancel',request_id,current,None,None))
        return True

    def events(self,request_id,ticket):
        try:
            while True:
                try:event=ticket.events.get(timeout=self.wait_seconds)
                except queue.Empty:raise TimeoutError('generation timed out') from None
                if 'error' in event:raise RuntimeError(event['error'])
                yield event
                if event['finished']:return
        finally:self.cancel(request_id,ticket=ticket)

    def _deliver(self,ticket,event):
        request_id=event['id']
        if event['finished']:
            self.engine.cancel(request_id)
            self._finish(request_id,ticket)
        try:ticket.events.put_nowait(event)
        except queue.Full:
            while True:
                try:ticket.events.get_nowait()
                except queue.Empty:break
            if not event['finished']:
                self.engine.cancel(request_id)
                self._finish(request_id,ticket)
            terminal=event if 'error' in event else dict(id=request_id,tokens=[],finished=True,error='client delivery queue exceeded capacity')
            ticket.events.put_nowait(terminal)
            return False
        return True

    def _command(self,command):
        kind,request_id,ticket,prompt,kwargs=command
        with self._lock:
            if self._tickets.get(request_id) is not ticket:return
        if kind=='cancel' or ticket.cancelled.is_set():
            ticket.error=RuntimeError('request cancelled')
            self._deliver(ticket,dict(id=request_id,tokens=[],finished=True,error='request cancelled'))
            ticket.ready.set();return
        try:self.engine.submit(request_id,prompt,**kwargs)
        except Exception as exc:
            self.engine.cancel(request_id)
            ticket.error=exc;self._finish(request_id,ticket)
        finally:ticket.ready.set()

    def _run(self):
        try:
            if self.context_factory is not None:context=self.context_factory()
            else:
                import mlx.core as mx
                context=mx.stream(mx.gpu if self.device=='metal' else mx.cpu)
            with context:
                while not self._stopped.is_set():
                    try:self._command(self._commands.get(timeout=.002 if self.engine.requests else .05))
                    except queue.Empty:pass
                    for _ in range(self.engine.max_requests):
                        try:command=self._commands.get_nowait()
                        except queue.Empty:break
                        self._command(command)
                        del command
                    with self._lock:generations=dict(self._tickets)
                    for event in self.engine.step() if self.engine.requests else ():
                        ticket=generations.get(event['id'])
                        with self._lock:current=self._tickets.get(event['id'])
                        if ticket is None or current is not ticket:continue
                        if ticket.cancelled.is_set():
                            event=dict(id=event['id'],tokens=[],finished=True,error='request cancelled')
                        self._deliver(ticket,event)
        except BaseException as exc:
            self._failure=exc
        finally:
            self._stopped.set()
            with self._lock:pending=list(self._tickets.items())
            for request_id,ticket in pending:
                ticket.error=RuntimeError(str(getattr(self,'_failure','service stopped')))
                self._deliver(ticket,dict(id=request_id,tokens=[],finished=True,error=str(ticket.error)))
                ticket.ready.set()

    def close(self):
        self._stopped.set()
        self._worker.join(timeout=10)
        if self._worker.is_alive():raise TimeoutError('model execution thread has not stopped')
