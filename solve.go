package mesh

import "math"

type (
	// LU Solver
	Solver interface {
		L(LU Mesher)
		U(LU Mesher)
		LU(mutate []int, L, U Mesher)
		Solve(A, B Mesher)
		SolveLy(b []float64)
		SolveUx(b []float64)
	}
)

// Solve Ly = b for y, replace b with solution
// column oriented forward substitution
func (L mesh) SolveLy(b []float64) {
	n := L.c
	for j := 0; j < n-1; j++ {
		b[j] /= L.Get(j, j)
		for k := j + 1; k < n; k++ {
			b[k] -= b[j] * L.Get(k, j)
		}
	}
	b[n-1] /= L.Get(n-1, n-1)
}

// Solve Ux = y, replace b with solution
// column oriented backward substitution
func (U mesh) SolveUx(y []float64) {
	n := U.c
	for j := n - 1; j >= 1; j-- {
		y[j] /= U.Get(j, j)
		for k := 0; k < j; k++ {
			y[k] -= y[j] * U.Get(k, j)
		}
	}
	y[0] /= U.Get(0, 0)
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
			aij := ACpy.Get(i, j)
			sum := 0.
			for k := 0; k <= i-1; k++ {
				aik := ACpy.Get(i, k)
				akj := ACpy.Get(k, j)
				sum += aik * akj
			}
			aij -= sum
			ACpy.Set(aij, i, j)
		}
		p := math.Abs(ACpy.Get(j, j))
		n := j
		for i := j + 1; i <= N-1; i++ {
			aij := ACpy.Get(i, j)
			sum := 0.
			for k := 0; k <= j-1; k++ {
				aik := ACpy.Get(i, k)
				akj := ACpy.Get(k, j)
				sum += aik * akj
			}
			aij -= sum
			ACpy.Set(aij, i, j)
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
			aij := ACpy.Get(i, j)
			ajj := ACpy.Get(j, j)
			aij /= ajj
			ACpy.Set(aij, i, j)
		}
	}
	L.L(ACpy)
	U.U(ACpy)
}

// L extracts Lower Triangular mesh from LU mesh
// and puts it in receiver
func (L *mesh) L(LU Mesher) {
	_, N := LU.Size()
	for j := 0; j < N; j++ {
		for i := j + 1; i < N; i++ {
			L.Set(LU.Get(i, j), i, j)
		}
	}
}

// U extracts Upper Triangular mesh from LU mesh
// and puts it in receiver
func (U *mesh) U(LU Mesher) {
	_, N := LU.Size()
	for j := 0; j < N; j++ {
		for i := 0; i <= j; i++ {
			U.Set(LU.Get(i, j), i, j)
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
	_, N := A.Size()
	br, bc := B.Size()
	mut := make([]int, N, N)
	b := make([]float64, br)
	L := I(N)
	U := Gen(nil, N, N)

	A.LU(mut, L, U)
	for j := 0; j < bc; j++ {
		B.GetCol(b, j)
		permute(b, mut)
		L.SolveLy(b)
		U.SolveUx(b)
		X.SetCol(b, j)
	}
}

// Permute swaps the elements in b with
// elements in b at mut[i] positions.
// mut vector contains a list of positions.
func permute(b []float64, mut []int) {
	for k := 0; k < len(b); k++ {
		if k != mut[k] && mut[k] != 0 {
			b[k], b[mut[k]] = b[mut[k]], b[k]
		}
	}
}
