package nn

import "gonum.org/v1/gonum/mat"

type Module interface {
	Forward(x *mat.Dense) *mat.Dense
}
