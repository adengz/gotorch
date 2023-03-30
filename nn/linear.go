package nn

import "gonum.org/v1/gonum/mat"

type linear struct {
	weight *mat.Dense
	bias   *mat.Dense
}

func (l *linear) Forward(x *mat.Dense) *mat.Dense {
	r, _ := x.Dims()
	c, _ := l.weight.Dims()
	y := mat.NewDense(r, c, nil)
	y.Mul(x, l.weight.T())
	if l.bias != nil {
		y.Add(y, l.bias)
	}
	return y
}

func Linear(weight, bias *mat.Dense) *linear {
	if weight == nil {
		panic("nil weight")
	}

	if bias != nil {
		rw, _ := weight.Dims()
		_, cb := bias.Dims()
		if rw != cb {
			panic("dimension not matching")
		}
	}

	return &linear{weight: weight, bias: bias}
}
