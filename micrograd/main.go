package main

import( 
	"fmt"
)

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

//func main() {
//	a := NewValue(3.0)
//	b := NewValue(3.0)
//	c := a.Add(b)
//	e := NewValue(3.0)
//	f := c.Mul(e)
//
//	f.backward()
//	f.PrettyPrint()
//}
//


// mainloop

func main() {
	// tiny dataset: 3 inputs -> 1 target, want the net to fit these points
	xs := [][]float64{
		{2, 3, -1},
		{3, -1, 0.5},
		{0.5, 1, 1},
		{1, 1, -1},
	}
	ys := []float64{1, -1, -1, 1}

	net := NewMLP(3, []int{4, 4, 1})
	learningRate := 0.001

	for epoch := 0; epoch < 200; epoch++ {
		// forward pass over the whole dataset, accumulate squared error
		loss := NewValue(0.0)
		for i, xrow := range xs {
			x := make([]*Value, len(xrow))
			for j, xi := range xrow {
				x[j] = NewValue(xi)
			}
			pred := net.Forward(x)[0]
			target := NewValue(ys[i])
			diff := pred.Add(target.Mul(NewValue(-1))) // pred - target (no Sub op, so *-1 then Add)
			loss = loss.Add(diff.Mul(diff))             // loss += diff^2
		}

		for _, p := range net.Parameters() {
			p.Grad = 0 // zero gradients before each backward pass
		}
		loss.backward()

		for _, p := range net.Parameters() {
			p.Data -= learningRate * p.Grad // gradient descent step
		}

		if epoch%20 == 0 {
			fmt.Printf("epoch %3d  loss=%.4f\n", epoch, loss.Data)
		}
	}

	fmt.Println("\nfinal predictions:")
	for i, xrow := range xs {
		x := make([]*Value, len(xrow))
		for j, xi := range xrow {
			x[j] = NewValue(xi)
		}
		pred := net.Forward(x)[0]
		fmt.Printf("  input=%v  target=%.1f  pred=%.4f\n", xrow, ys[i], pred.Data)
	}
}

