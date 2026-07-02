package main
import "math/rand"

// ---------- nn: neuron / layer / mlp ----------
// No activation function on purpose: tanh/relu need more than add+mul
// (exp, or a max/comparison op), which we deliberately left out. That means
// stacking Layers is still just one big linear function under the hood --
// fine for seeing how gradients flow through weights and biases across
// several layers, but it won't fit anything a single linear layer couldn't.

var rng = rand.New(rand.NewSource(42))

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := range w {
		w[i] = NewValue(rng.Float64()*2 - 1) // random in [-1, 1]
	}
	return &Neuron{
		Weights: w,
		Bias:    NewValue(rng.Float64()*2 - 1),
	}
}

func (n *Neuron) Forward(x []*Value) *Value {
	out := n.Bias
	for i, w := range n.Weights {
		out = out.Add(w.Mul(x[i]))
	}
	return out
}

func (n *Neuron) Parameters() []*Value {
	return append(append([]*Value{}, n.Weights...), n.Bias)
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := range neurons {
		neurons[i] = NewNeuron(nin)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(x)
	}
	return out
}

func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

type MLP struct {
	Layers []*Layer
}

// nin is the input size, nouts is the size of each layer after that.
// e.g. NewMLP(3, []int{4, 4, 1}) => 3 inputs -> hidden(4) -> hidden(4) -> 1 output
func NewMLP(nin int, nouts []int) *MLP {
	sizes := append([]int{nin}, nouts...)
	layers := make([]*Layer, len(nouts))
	for i := range layers {
		layers[i] = NewLayer(sizes[i], sizes[i+1])
	}
	return &MLP{Layers: layers}
}

func (m *MLP) Forward(x []*Value) []*Value {
	for _, layer := range m.Layers {
		x = layer.Forward(x)
	}
	return x
}

func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}
