package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type scalarActivationFunc func(x float64) float64

type scalarActivation struct {
	f scalarActivationFunc
}

func (s *scalarActivation) Forward(x *mat.Dense) *mat.Dense {
	x.Apply(func(_, _ int, v float64) float64 {
		return s.f(v)
	}, x)
	return x
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt(2)))
}

func Sigmoid() *scalarActivation {
	return &scalarActivation{f: sigmoid}
}

func GELU() *scalarActivation {
	return &scalarActivation{f: gelu}
}

type softmax struct{}

func (s softmax) Forward(x *mat.Dense) *mat.Dense {
	x.Apply(func(_, _ int, v float64) float64 {
		return math.Exp(v)
	}, x)
	sum := mat.Sum(x)
	x.Scale(1/sum, x)
	return x
}

func Softmax() *softmax {
	return &softmax{}
}
