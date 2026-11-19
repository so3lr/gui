addToLibrary({
 ring_drain: (p,c) => Module['proactor'].ring_drain(p,c),
 poll_wait: t => Module['proactor'].poll_wait(t),
 timer_start: (a,b,c) => Module['proactor'].timer_start(a,b,c),
 timer_cancel: a => Module['proactor'].timer_cancel(a),
 wake: () => Module['proactor'].wake(),
 schedule_drain: () => Module['proactor'].schedule_drain()
});
