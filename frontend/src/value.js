/**
 * JS port of the Value autograd class from microgpt.py.
 * Stores a scalar value and its gradient as a node in a computation graph.
 * Uses explicit stack for topological sort to handle ~48k nodes without stack overflow.
 */

export class Value {
  constructor(data, children, localGrads) {
    this.data = data;
    this.grad = 0;
    this._children = children || null;
    this._localGrads = localGrads || null;
  }

  add(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return new Value(this.data + other.data, [this, other], [1, 1]);
  }

  mul(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }

  pow(n) {
    return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
  }

  log() {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp() {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu() {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return this.add(other.neg());
  }

  div(other) {
    if (!(other instanceof Value)) other = new Value(other);
    return this.mul(other.pow(-1));
  }

  backward() {
    // Topological sort with explicit stack (no recursion)
    const topo = [];
    const visited = new Set();
    const stack = [{ node: this, phase: 0 }];

    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const v = frame.node;

      if (visited.has(v)) {
        stack.pop();
        continue;
      }

      if (!v._children || frame.phase >= v._children.length) {
        visited.add(v);
        topo.push(v);
        stack.pop();
      } else {
        const child = v._children[frame.phase];
        frame.phase++;
        if (!visited.has(child)) {
          stack.push({ node: child, phase: 0 });
        }
      }
    }

    // Backward pass
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      if (!v._children) continue;
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}
