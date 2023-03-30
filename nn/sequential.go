package nn

import "gonum.org/v1/gonum/mat"

type Sequential []Module

func (s Sequential) Forward(x *mat.Dense) *mat.Dense {
	for _, m := range s {
		x = m.Forward(x)
	}
	return x
}
