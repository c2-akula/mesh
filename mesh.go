package mesh

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/c2-akula/mesh/amd"
)

const (
	Tol = 2.2737e-13
	Eps = 2.220446049250313e-16
)

const (
	Down = dir(iota)
	Right
)

const (
	One = norm(iota) + 1
	Two
	Inf
)

type (
	dir  uint8
	norm uint8
	mesh struct {
		elems []float64
		r, c  int
	}

	Powerer interface {
		Expm(m Mesher)
		Powm(m Mesher, s int)
	}

	Arither interface {
		Sum(a, b float64, m, n Mesher)
		Mul(a float64, m, n Mesher)
		Norm(ord norm) float64
		Inv(m Mesher)
	}

	// Infoer lists behaviors to extract information
	// from a mesh
	Infoer interface {
		Det() (d float64)
		Get(i, j int) float64
		GetCol(v []float64, c int)
		GetDiag(d []float64)
		GetRow(v []float64, r int)
		Size() (r, c int)
		Slice() []float64
		IsSquare() bool
		IsSymmetric() bool
	}

	Trianguler interface {
		Triu(m Mesher, k int)
		Tril(m Mesher, k int)
	}

	Mesher interface {
		Arither
		Infoer
		Manipulator
		Powerer
		Solver
		Trianguler
		Submesh(m Mesher, i, j int)
	}

	Manipulator interface {
		// Copy from mesh into to mesh
		Clone(frm Mesher)
		Scale(a float64)
		Set(a float64, i, j int)
		SetCol(v []float64, c int)
		SetDiag(d []float64)
		SetMesh(m Mesher, i, j int)
		SetRow(v []float64, r int)
		Stack(m, n Mesher, d dir)
		SwapCols(c1, c2 int)
		SwapRows(r1, r2 int)
		T(m Mesher)
		Zero()
	}
)

// Gen creates a new row oriented mesh
// If nil vector is passed, then mesh
// will be generated with all elements zeroed
func Gen(v []float64, r, c int) Mesher {
	m := new(mesh)
	m.r, m.c = r, c
	m.elems = make([]float64, r*c)
	if v == nil {
		return m
	}
	copy(m.elems, v)
	return m
}

// Random generates a random rxc mesh
func Random(r, c int) Mesher {
	rm := Gen(nil, r, c)
	el := rm.(*mesh).elems
	for j := range el {
		el[j] = rand.Float64()
	}
	return rm
}

// Clone copies 'frm' mesh into 'to'
// mesh
func (to *mesh) Clone(frm Mesher) {
	copy(to.elems, frm.(*mesh).elems)
}

func (m mesh) Slice() []float64 {
	return m.elems
}

// I creates an identity mesh.
func I(n int) Mesher {
	eye := Gen(nil, n, n)
	for e := 0; e < n; e++ {
		eye.Set(1, e, e)
	}
	return eye
}

func (m mesh) String() string {
	s := ""
	r, c := m.r, m.c
	elems := m.elems
	for i := 0; i < r; i++ {
		s += "{"
		tmp := i * c
		for j := 0; j < c; j++ {
			s += fmt.Sprintf(" %.3g ", elems[tmp+j])
		}
		s += "}\n"
	}
	return s
}

// Submesh gets a submesh of dims rxc starting at i,j from m and put in n
func (n *mesh) Submesh(m Mesher, i, j int) {
	mr, mc := m.Size()
	nr, nc := n.Size()
	// we check if the size of mesh, n is either
	// equal to or less than the mesh m, from which
	// we want to extract a mesh of size(n)
	nelems := n.elems
	melems := m.(*mesh).elems
	if i+nr <= mr && j+nc <= mc {
		for p := 0; p < nr; p++ {
			ptmp := p * nc
			iptmp := (i+p)*mc + j
			for q := 0; q < nc; q++ {
				nelems[ptmp+q] = melems[iptmp+q]
			}
		}
	} else {
		panic("mesh: Submesh: Size of submesh too big for given indices!")
	}
}

// Size returns the rows, columns and
// stride of the mesh
func (m mesh) Size() (r, c int) {
	r, c = m.r, m.c
	return
}

// Get gets an element at node (i,j)
func (m *mesh) Get(i, j int) float64 {
	return m.elems[i*m.c+j]
}

// Set sets an element, a at node (i,j)
func (m *mesh) Set(a float64, i, j int) {
	m.elems[i*m.c+j] = a
}

// Mul computes the product of two matrices a,b and puts
// it in receiver.
// a = mxk, b = kxn, c = mxn
func (c *mesh) Mul(s float64, a, b Mesher) {
	m, k := a.Size()
	br, n := b.Size()

	var (
		icoff, iaoff, lboff int
		ctmp                []float64
		tmp                 float64
	)

	if k != br {
		panic("mesh: Mul: Columns of 'a' != Rows of 'b'")
	}

	aoff, boff, coff := k, n, n
	for i := 0; i < m; i++ {
		icoff = i * coff
		iaoff = i * aoff
		ctmp = c.elems[icoff : icoff+n]
		for l, v := range a.(*mesh).elems[iaoff : iaoff+k] {
			lboff = l * boff
			tmp = s * v
			if tmp != 0 {
				amd.DaxpyUnitary(tmp, b.(*mesh).elems[lboff:lboff+n], ctmp, ctmp)
			}
		}
	}
}

// GetDiag puts the diagonal of the mesh into
// the vector, m must be a square matrix
func (m mesh) GetDiag(d []float64) {
	for e := range d {
		d[e] = m.Get(e, e)
	}
}

// SetDiag puts the diagonal, d into the mesh
// m must be a square matrix
func (m *mesh) SetDiag(d []float64) {
	for e := range d {
		m.Set(d[e], e, e)
	}
}

// GetRow returns the r'th column of m into vector, v
// of length equal to no. of cols of mesh
func (m mesh) GetRow(v []float64, r int) {
	off := r * m.c
	for j := range v {
		v[j] = m.elems[off+j]
	}
}

// GetCol returns the c'th column of m into vector, v
// of length equal to no. of rows of mesh
func (m mesh) GetCol(v []float64, c int) {
	off := m.c
	for i := range v {
		v[i] = m.elems[i*off+c]
	}
}

// SetRow sets the r'th column of m with vector, v
func (m *mesh) SetRow(v []float64, r int) {
	off := r * m.c
	for j, e := range v {
		m.elems[off+j] = e
	}
}

// SetCol sets the c'th column of m with vector, v
func (m *mesh) SetCol(v []float64, c int) {
	for i, e := range v {
		m.elems[i*m.c+c] = e
	}
}

// SwapCols swaps cols c1, c2 in mesh
func (m *mesh) SwapCols(c1, c2 int) {
	_, mc := m.Size()
	mvec := m.elems
	for j := 0; j < mc; j++ {
		mvec[j*mc+c1], mvec[j*mc+c2] = mvec[j*mc+c2], mvec[j*mc+c1]
	}
}

// SwapRows swaps rows r1, r2 in mesh
func (m *mesh) SwapRows(r1, r2 int) {
	mr, mc := m.Size()
	mvec := m.elems
	r1off, r2off := r1*mc, r2*mc
	for i := 0; i < mr; i++ {
		mvec[r1off+i], mvec[r2off+i] = mvec[r2off+i], mvec[r1off+i]
	}
}

// SetMesh copies part of the mesh, m into receiver, starting at (i, j)
func (n *mesh) SetMesh(m Mesher, i, j int) {
	mr, mc := m.Size()
	for r := 0; r < mr; r++ {
		for c := 0; c < mc; c++ {
			n.Set(m.Get(r, c), i+r, j+c)
		}
	}
}

// Stack stacks mesh, m, Down or Right of the mesh, n
// and puts the result in receiver
func (o *mesh) Stack(m, n Mesher, d dir) {
	switch d {
	case Down:
		o.SetMesh(m, 0, 0)
		o.SetMesh(n, m.(*mesh).r, 0)
	case Right:
		o.SetMesh(m, 0, 0)
		o.SetMesh(n, 0, m.(*mesh).c)
	}
}

// T transposes mesh, m and puts it into
// the receiver
func (n *mesh) T(m Mesher) {
	mr, mc := m.Size()
	nr, _ := n.Size()
	if mc != nr {
		panic("mesh: T: Cols of 'm' != Rows of 'n'")
	}
	for i := 0; i < mr; i++ {
		for j := 0; j < mc; j++ {
			n.Set(m.Get(i, j), j, i)
		}
	}
}

// Tip transposes a mesh in-place
func (m *mesh) Tip() {
	h, moff := m.Size()
	for start := range m.elems {
		next := start
		i := 0
		for {
			i++
			next = (next%h)*moff + next/h
			if next <= start {
				break
			}
		}

		if next < start || i == 1 {
			continue
		}

		next = start
		tmp := m.elems[next]
		for {
			i = (next%h)*moff + next/h
			if i == start {
				m.elems[next] = tmp
			} else {
				m.elems[next] = m.elems[i]
			}
			next = i
			if next <= start {
				break
			}
		}
	}
	m.r, m.c = m.c, m.r
}

// Triu returns the elements on and above the kth diagonal of 'm'.
// k = 0 is the main diagonal
// k > 0 is above the main diagonal
// k < 0 is below the main diagonal
func (n *mesh) Triu(m Mesher, k int) {
	nr, nc := m.Size()
	if k > 0 && k > nc || k < 0 && k < -nr {
		panic("mesh: Triu: requested diagonal out of range.")
	}

	for j := int(math.Max(0, float64(k))); j < nc; j++ {
		nr_lim := int(math.Min(float64(nr), float64(j-k)))
		for l := 0; l < nr_lim+1; l++ {
			n.Set(m.Get(l, j), l, j)
		}
	}
}

// Tril returns the elements on and below the kth diagonal of 'm'.
// k = 0 is the main diagonal
// k > 0 is above the main diagonal
// k < 0 is below the main diagonal
func (n *mesh) Tril(m Mesher, k int) {
	nr, nc := m.Size()
	if k > 0 && k > nc || k < 0 && k < -nr {
		panic("mesh: Tril: requested diagonal out of range.")
	}

	for j := 0; j < int(math.Min(float64(nc), float64(nr+k))); j++ {
		nr_lim := int(math.Max(0, float64(j-k)))
		for l := nr_lim; l < nr; l++ {
			n.Set(m.Get(l, j), l, j)
		}
	}
}

func (m *mesh) IsSquare() bool {
	return m.r == m.c
}

func (m *mesh) IsSymmetric() bool {
	for i := 0; i < m.r; i++ {
		for j := 0; j < i; j++ {
			if m.Get(i, j) != m.Get(j, i) {
				return false
			}
		}
	}
	return true
}

// Norm computes the magnitude of the mesh.
// Ord can be One, Two or Inf
func (m mesh) Norm(ord norm) float64 {
	switch ord {
	case One:
		sv := make([]float64, m.c) // hold the sums of each col

		// we extract each column
		for j := 0; j < m.c; j++ {
			// we extract the elems in the col
			sum := 0. // we wanna keep track of the sum of each col
			for k := 0; k < m.r; k++ {
				sum += math.Abs(m.elems[k*m.c+j])
			}
			// we then wanna put it inside a vector so that
			// we can get the biggest col sum out
			sv[j] = sum
		}

		// return the biggest column sum
		_, max := VecMax(sv)
		return max
	case Two:
		sum := 0.
		for _, v := range m.elems {
			sum += v * v
		}
		return math.Sqrt(sum)
	case Inf:
		sv := make([]float64, m.c) // no. of elems in row = no. of cols

		// we extract each row
		for i := 0; i < m.r; i++ {
			// we extract the elems in the row
			sum := 0. // we wanna keep track of the sum of each row
			for j := 0; j < m.c; j++ {
				sum += math.Abs(m.elems[i*m.c+j])
			}
			// we wanna put the sum into a vector so that
			// we can get the biggest row sum out
			sv[i] = sum
		}

		// return the biggest row sum
		_, max := VecMax(sv)
		return max
	}
	return 0
}

// Scale multiplies a mesh with a scalar
func (m *mesh) Scale(a float64) {
	switch a {
	case 0:
		for k := 0; k < len(m.elems); k++ {
			m.elems[k] = 0
		}
	case 1:
	default:
		for k := 0; k < len(m.elems); k++ {
			m.elems[k] *= a
		}
	}
}

// Inv computes the inverse of m into n
func (n *mesh) Inv(m Mesher) {
	eye := I(m.(*mesh).c)
	n.Solve(m, eye)
}

// Det computes the determinant of a square mesh, m
func (m mesh) Det() (det float64) {
	mc := m.c
	if mc == 3 {
		a := m.Get(0, 0)
		e := m.Get(1, 1)
		i := m.Get(2, 2)

		b := m.Get(0, 1)
		f := m.Get(1, 2)
		g := m.Get(2, 0)

		c := m.Get(0, 2)
		d := m.Get(1, 0)
		h := m.Get(2, 1)

		det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
		return
	} else if mc == 2 {
		a := m.Get(0, 0)
		d := m.Get(1, 1)

		b := m.Get(0, 1)
		c := m.Get(1, 0)

		det = a*d - b*c
		return
	}

	L := I(mc)
	U := Gen(nil, mc, mc)
	mut := make([]int, mc)
	m.LU(mut, L, U)
	det = 1.

	// check number of mutations
	// if number of mutations is
	// odd, then det is -det
	nmuts := func(det *float64) {
		cnt := 0
		for _, val := range mut {
			if val != 0 {
				cnt++
			}
		}

		if cnt%2 == 1 {
			*det = -*det
		}
	}

	ld := make([]float64, mc)
	ud := make([]float64, mc)

	L.GetDiag(ld)
	U.GetDiag(ud)

	detl, detu := 1., 1.
	for k, v := range ld {
		detl *= v
		detu *= ud[k]
	}

	det = detl * detu
	nmuts(&det)
	return
}

// Sum computes the sum of meshes, m & n and puts
// result in receiver.
// m, n and o should have same dimensions.
// a*m + n = o
// we will calculate the sum of meshes using
// a slight variation of saxpy
// ie ax+by, x and y are vectors and
// a, b are scalars.
func (o *mesh) Sum(a, b float64, m, n Mesher) {

	// the size of all the meshes are assumed to be
	// equal and no check is performed, we want this
	// to be as fast an operation as possible
	_, mc := m.Size() // no. of cols of the meshes

	mv := make([]float64, mc) // col vector of m
	nv := make([]float64, mc) // col vector of n
	ov := make([]float64, mc) // col vector of o

	// we will scale the meshes
	if a == 0 && b == 1 {
		// z = 0*x + y
		// we just copy n into o
		o.Clone(n)
		return
	}

	if b == 0 && a == 1 {
		// z = x + 0*y
		// we just copy m into o
		o.Clone(m)
		return
	}

	if a == 1 && b != 1 {
		// z = x + by
		// we scale n
		ns := n.Slice()
		for j := range ns {
			ns[j] *= b
		}
	} else if a != 1 && b == 1 {
		// z = ax + y
		// we scale m
		ms := m.Slice()
		for j := range ms {
			ms[j] *= a
		}
	} else if a == 0 && b == 0 {
		o.Zero()
		return
	}

	// we add the meshes by cols
	for j := 0; j < mc; j++ {
		m.GetCol(mv, j)
		n.GetCol(nv, j)
		amd.DaxpyUnitary(a, mv, nv, ov)
		o.SetCol(ov, j)
	}
}

// Zero zeroes the mesh for reuse
func (m *mesh) Zero() {
	for k := range m.elems {
		m.elems[k] = 0
	}
}

// IsEqual tells if two meshes are equal
func (n mesh) IsEqual(m Mesher) (iseq bool) {
	for k, v := range n.elems {
		if v != m.(*mesh).elems[k] {
			return
		}
	}
	iseq = true
	return
}
