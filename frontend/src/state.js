const state = {};
const listeners = {};

export function get(key) {
  return state[key];
}

export function set(key, value) {
  if (state[key] === value) return;
  state[key] = value;
  for (const fn of listeners[key] || []) fn(value);
}

export function subscribe(key, fn) {
  (listeners[key] ??= []).push(fn);
  return () => {
    listeners[key] = listeners[key].filter(f => f !== fn);
  };
}
