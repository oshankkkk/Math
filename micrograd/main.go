package main

import "fmt"

type Value struct {
	Data       float64
	Grad       float64
	Prev       []*Value
	Op         string
	backwardFn func()
}

func NewValue(data float64) *Value {
	return &Value{Data: data}
}

func (v *Value) Add(other *Value) *Value {
	out := &Value{
		Data: v.Data + other.Data,
		Prev: []*Value{v, other},
		Op:   "+",
	}
	out.backwardFn = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := &Value{
		Data: v.Data * other.Data,
		Prev: []*Value{v, other},
		Op:   "*",
	}
	
	out.backwardFn = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

func (v *Value) backward() {
	var topo []*Value
	visited := make(map[*Value]bool)

	var buildTopo func(*Value)
	buildTopo = func(n *Value) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, child := range n.Prev {
			buildTopo(child)
		}
		topo = append(topo, n)
	}
	buildTopo(v)

	// dL/dL = 1, final node deravitive
	v.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backwardFn != nil {
			topo[i].backwardFn()
		}
	}
}

func (v *Value) PrettyPrint() {
	v.printNode(0)
}

func (v *Value) printNode(depth int) {
	op := v.Op
	if op == "" {
		op = "leaf"
	}
	indent := ""
	for i := 0; i < depth; i++ {
		indent += "  "
	}

	fmt.Printf("%sValue(data=%.4f, grad=%.4f, op=%s)\n", indent, v.Data, v.Grad, op)
	for _, child := range v.Prev {
		child.printNode(depth + 1)
	}
}

func main() {
	a := NewValue(3.0)
	b := NewValue(3.0)
	c := a.Add(b)
	e := NewValue(3.0)
	f := c.Mul(e)

	f.backward()
	f.PrettyPrint()
}
