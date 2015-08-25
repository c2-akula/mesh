/*
   package vec provides some common vector routines
   like cross and dot products, creating skew-symmetric
   meshes etc.
*/
package vec

import (
	"log"
	"math"
	"math/rand"

	"github.com/c2-akula/mesh"
	"github.com/c2-akula/mesh/amd"
)

// Random creates a random valued vector with entries
// between [0, 1)
func Random(v []float64) {
	for j := range v {
		v[j] = rand.Float64()
	}
}

// Skew constructs a skew-symmetric matrix from
// 3x1 mesh, w into 3x3 mesh, r
// The size of w and r is not checked since
// we want this routine to return
// as soon as possible
func Skew(w, r mesh.Mesher) {
	w0, w1, w2 := w.Get(0, 0), w.Get(1, 0), w.Get(2, 0)
	r.Set(-w2, 0, 1)
	r.Set(w1, 0, 2)
	r.Set(w2, 1, 0)
	r.Set(-w0, 1, 2)
	r.Set(-w1, 2, 0)
	r.Set(w0, 2, 1)
}

// IsEqual is a lower level routine that checks
// if two vectors are equal
func IsEqual(a, b []float64) (eq bool) {
	for k, v := range b {
		if a[k] != v {
			log.Println("mesh/vec: IsEqual failed. lhs, rhs: ", a[k], v)
			return
		}
	}
	eq = true
	return
}

// Clear zeros vector a
func Clear(a []float64) {
	mesh.Clear(a)
}

// Scale multiplies vector, 'a' with scalar 's'
func Scale(s float64, a []float64) {
	// for s == 1 we do nothing
	if s != 0 && s != 1 {
		for e := range a {
			a[e] *= s
		}
		return
	}

	if s == 0 {
		Clear(a)
	}
}

// ElemSum computes the sum of all the elements in
// a vector
func ElemSum(v []float64, isabs bool) (sum float64) {
	if isabs {
		for _, e := range v {
			sum += math.Abs(e)
		}
		return
	}

	for _, e := range v {
		sum += e
	}
	return
}

// Norm computes the vector norm sqrt(x^2 + y^2 + ...)
func Norm(v []float64) (n float64) {
	for _, e := range v {
		n += e * e
	}
	n = math.Sqrt(n)
	return
}

// Dot computes the dot product of two vectors
// d = xi*yi + xj*yj + xk*yk
func Dot(a, b []float64) (d float64) {
	d = amd.DdotUnitary(a, b)
	return
}

// Dot3 computes the dot product of two vectors of len = 3
func Dot3(a, b []float64) (d float64) {
	d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
	return
}

// Dot4 computes the dot product of two vectors of len = 4
func Dot4(a, b []float64) (d float64) {
	d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
	return
}

// Dot6 computes the dot product of two vectors of len = 6
func Dot6(a, b []float64) (d float64) {
	d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5]
	return
}

// Cross computes the vector product of two vectors, a & b
// and puts the result in c. a, b, c are all 3x1 vectors.
func Cross(a, b, c []float64) {
	// we don't check the length of the vectors
	// since we want to return as fast as possible
	a0, a1, a2 := a[0], a[1], a[2]
	b0, b1, b2 := b[0], b[1], b[2]
	c[0] = a1*b2 - a2*b1
	c[1] = a2*b0 - a0*b2
	c[2] = a0*b1 - a1*b0
}

// Sum computes the sum of two vectors
func Sum(s float64, a, b, c []float64) {
	amd.DaxpyUnitary(s, a, b, c)
}

// Sum3 computes the sum of two vectors, a & b of len = 3
func Sum3(s float64, a, b, c []float64) {
	c[0] = s*a[0] + b[0]
	c[1] = s*a[1] + b[1]
	c[2] = s*a[2] + b[2]
}

// Sum4 computes the sum of two vectors, a & b of len = 4
func Sum4(s float64, a, b, c []float64) {
	c[0] = s*a[0] + b[0]
	c[1] = s*a[1] + b[1]
	c[2] = s*a[2] + b[2]
	c[3] = s*a[3] + b[3]
}

// Sum6 computes the sum of two vectors, a & b of len = 6
func Sum6(s float64, a, b, c []float64) {
	c[0] = s*a[0] + b[0]
	c[1] = s*a[1] + b[1]
	c[2] = s*a[2] + b[2]
	c[3] = s*a[3] + b[3]
	c[4] = s*a[4] + b[4]
	c[5] = s*a[5] + b[5]
}
