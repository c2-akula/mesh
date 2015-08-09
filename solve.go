package mesh

import (
	"github.com/vec"
	"math"
)

type (
	// LU Solver
	Solver interface {
		L(LU Mesher)
		U(LU Mesher)
		LU(mutate []int, L, U Mesher)
		Solve(A, B Mesher)
		SolveLy(b vec.Vectorer)
		SolveUx(b vec.Vectorer)
	}
)

// Solve Ly = b for y, replace b with solution
// column oriented forward substitution
func (L mesh) SolveLy(b vec.Vectorer) {
	n := L.c
	for j := 0; j < n-1; j++ {
		b.SetAt(j, b.GetAt(j)/L.GetAtNode(j, j))
		for k := j + 1; k < n; k++ {
			b.SetAt(k, b.GetAt(k)-b.GetAt(j)*L.GetAtNode(k, j))
		}
	}
	b.SetAt(n-1, b.GetAt(n-1)/L.GetAtNode(n-1, n-1))
}

// Solve Ux = y, replace b with solution
// column oriented backward substitution
func (U mesh) SolveUx(y vec.Vectorer) {
	n := U.c
	for j := n - 1; j >= 1; j-- {
		y.SetAt(j, y.GetAt(j)/U.GetAtNode(j, j))
		for k := 0; k < j; k++ {
			y.SetAt(k, y.GetAt(k)-y.GetAt(j)*U.GetAtNode(k, j))
		}
	}
	y.SetAt(0, y.GetAt(0)/U.GetAtNode(0, 0))
}

// http://ezekiel.vancouver.wsu.edu/~cs330/lectures/linear_algebra/LU.pdf
// Crout's Method with Partial Pivoting
// mutate should be length N
// L, U should be square
func (A mesh) LU(mutate []int, L, U Mesher) {
	ACpy := Gen(nil, A.r, A.c)
	ACpy.Clone(&A)
	N := A.c
	d := 1
	for j := 0; j < N; j++ {
		for i := 0; i <= j; i++ {
			aij := ACpy.GetAtNode(i, j)
			sum := 0.
			for k := 0; k <= i-1; k++ {
				aik := ACpy.GetAtNode(i, k)
				akj := ACpy.GetAtNode(k, j)
				sum += aik * akj
			}
			aij -= sum
			ACpy.SetAtNode(aij, i, j)
		}
		p := math.Abs(ACpy.GetAtNode(j, j))
		n := j
		for i := j + 1; i <= N-1; i++ {
			aij := ACpy.GetAtNode(i, j)
			sum := 0.
			for k := 0; k <= j-1; k++ {
				aik := ACpy.GetAtNode(i, k)
				akj := ACpy.GetAtNode(k, j)
				sum += aik * akj
			}
			aij -= sum
			ACpy.SetAtNode(aij, i, j)
			if math.Abs(aij) > p {
				p = math.Abs(aij)
				n = i
			}
		}
		if p == 0 {
			panic("mesh: LU: Singular mesh!")
		}

		if n != j {
			ACpy.SwapRows(n, j)
			mutate[n], mutate[j] = j, n
			d = -d
		}
		for i := j + 1; i <= N-1; i++ {
			aij := ACpy.GetAtNode(i, j)
			ajj := ACpy.GetAtNode(j, j)
			aij /= ajj
			ACpy.SetAtNode(aij, i, j)
		}
	}
	L.L(ACpy)
	U.U(ACpy)
	// A.L(L)
	// A.U(U)
}

// L extracts Lower Triangular mesh from LU mesh
// and puts it in receiver
func (L *mesh) L(LU Mesher) {
	_, N, _ := LU.Size()
	for j := 0; j < N; j++ {
		for i := j + 1; i < N; i++ {
			L.SetAtNode(LU.GetAtNode(i, j), i, j)
		}
	}
}

// U extracts Upper Triangular mesh from LU mesh
// and puts it in receiver
func (U *mesh) U(LU Mesher) {
	_, N, _ := LU.Size()
	for j := 0; j < N; j++ {
		for i := 0; i <= j; i++ {
			U.SetAtNode(LU.GetAtNode(i, j), i, j)
		}
	}
}

/*
	Solve AX = B, for X
	if A, B are square, then X should be square
	if A is square, but B, X are rectangular,
	then A.R == B.R == X.R else Solve will panic
*/
func (X *mesh) Solve(A, B Mesher) {
	_, N, _ := A.Size()
	br, bc, _ := B.Size()
	mut := make([]int, N, N)
	b := vec.Zeros(br)
	L := I(N)
	U := Gen(nil, N, N)

	A.LU(mut, L, U)
	for j := 0; j < bc; j++ {
		B.GetCol(b, j)
		b.Permute(mut)
		L.SolveLy(b)
		U.SolveUx(b)
		X.SetCol(b, j)
	}
}
