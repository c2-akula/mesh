package mesh

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/vec"
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
		elems     vec.Vectorer
		r, c, off int
	}

	Powerer interface {
		Expm(m Mesher)
		Powm(m Mesher, s int)
	}

	Arither interface {
		Sum(a float64, m, n Mesher)
		Mul(m, n Mesher)
		// NewExpm(m Mesher)
		// PExpm(m Mesher)
		Norm(ord norm) float64
		Inv(m Mesher)
	}

	// Infoer lists behaviors to extract information
	// from a mesh
	Infoer interface {
		Det() (d float64)
		GetAtNode(i, j int) float64
		GetCol(v vec.Vectorer, c int)
		GetDiag(d vec.Vectorer)
		GetRow(v vec.Vectorer, r int)
		Vec() vec.Vectorer
		Size() (r, c, off int)
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
		SetAtNode(a float64, i, j int)
		SetCol(v vec.Vectorer, c int)
		SetDiag(d vec.Vectorer)
		SetMesh(m Mesher, i, j int)
		SetRow(v vec.Vectorer, r int)
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
func Gen(v vec.Vectorer, r, c int) Mesher {
	hm := new(mesh)
	hm.r, hm.c, hm.off = r, c, c
	hm.elems = vec.Zeros(r * c)
	if v == nil {
		return hm
	}
	hm.elems.Clone(v)
	return hm
}

// Random generates a random rxc mesh
func Random(r, c int) Mesher {
	rm := Gen(nil, r, c)
	for j := range rm.Vec().Slice() {
		rm.Vec().SetAt(j, rand.Float64())
	}
	return rm
}

// Vec gives access the element vector
// of the mesh
func (m mesh) Vec() vec.Vectorer {
	return m.elems
}

// Clone copies 'frm' mesh into 'to'
// mesh
func (to *mesh) Clone(frm Mesher) {
	to.elems.Clone(frm.(*mesh).elems)
}

// I creates an identity mesh.
func I(n int) Mesher {
	eye := Gen(nil, n, n)
	for e := 0; e < n; e++ {
		eye.SetAtNode(1, e, e)

	}
	return eye
}

func (m mesh) String() string {
	s := ""
	r, c, off := m.r, m.c, m.off
	elems := m.elems.Slice()
	for i := 0; i < r; i++ {
		s += "{"
		tmp := i * off
		for j := 0; j < c; j++ {
			s += fmt.Sprintf(" %.3g ", elems[tmp+j])
		}
		s += "}\n"
	}
	return s
}

// Submesh gets a submesh of dims rxc starting at i,j from m and put in n
func (n *mesh) Submesh(m Mesher, i, j int) {
	mr, mc, moff := m.Size()
	nr, nc, noff := n.Size()
	// we check if the size of mesh, n is either
	// equal to or less than the mesh m, from which
	// we want to extract a mesh of size(n)
	nelems := n.elems.Slice()
	melems := m.(*mesh).elems.Slice()
	if i+nr <= mr && j+nc <= mc {
		for p := 0; p < nr; p++ {
			ptmp := p * noff
			iptmp := (i+p)*moff + j
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
func (m mesh) Size() (r, c, off int) {
	r, c, off = m.r, m.c, m.off
	return
}

// GetAtNode gets an element at node (i,j)
func (m *mesh) GetAtNode(i, j int) float64 {
	return m.elems.Slice()[i*m.off+j]
}

// SetAtNode sets an element, a at node (i,j)
func (m *mesh) SetAtNode(a float64, i, j int) {
	m.elems.Slice()[i*m.off+j] = a
}

// Mul computes the product of two matrices m,n and puts
// it in receiver.
// m = pxr, n = rxq, o = pxq
func (o *mesh) Mul(m, n Mesher) {
	mr, mc, _ := m.Size()
	nr, nc, _ := n.Size()
	or, oc, _ := o.Size()

	if mc != nr {
		panic("mesh: Mul: Input meshes dimension mismatch.")
	}

	if or != mr && oc != nc {
		panic("mesh: Mul: Receiver has incorrect dimensions.")
	}

	o.Vec().Mul(1, m.Vec(), n.Vec(), or, oc, mr)
}

// GetDiag puts the diagonal of the mesh into
// the vector, m must be a square matrix
func (m mesh) GetDiag(d vec.Vectorer) {
	for e := range d.Slice() {
		d.SetAt(e, m.GetAtNode(e, e))
	}
}

// SetDiag puts the diagonal, d into the mesh
// m must be a square matrix
func (m *mesh) SetDiag(d vec.Vectorer) {
	for e := range d.Slice() {
		m.SetAtNode(d.GetAt(e), e, e)
	}
}

// GetRow returns the r'th column of m into vector, v
// of length equal to no. of cols of mesh
func (m mesh) GetRow(v vec.Vectorer, r int) {
	off := r * m.off
	for j := range v.Slice() {
		v.SetAt(j, m.elems.GetAt(off+j))
	}
}

// GetCol returns the c'th column of m into vector, v
// of length equal to no. of rows of mesh
func (m mesh) GetCol(v vec.Vectorer, c int) {
	off := m.off
	for i := range v.Slice() {
		v.SetAt(i, m.elems.GetAt(i*off+c))
	}
}

// SetRow sets the r'th column of m with vector, v
func (m *mesh) SetRow(v vec.Vectorer, r int) {
	off := r * m.off
	for j, e := range v.Slice() {
		m.elems.SetAt(off+j, e)
	}
}

// SetCol sets the c'th column of m with vector, v
func (m *mesh) SetCol(v vec.Vectorer, c int) {
	for i, e := range v.Slice() {
		m.elems.SetAt(i*m.off+c, e)
	}
}

// SwapCols swaps cols c1, c2 in mesh
func (m *mesh) SwapCols(c1, c2 int) {
	_, mc, moff := m.Size()
	mvec := m.elems.Slice()
	for j := 0; j < mc; j++ {
		mvec[j*moff+c1], mvec[j*moff+c2] = mvec[j*moff+c2], mvec[j*moff+c1]
	}
}

// SwapRows swaps rows r1, r2 in mesh
func (m *mesh) SwapRows(r1, r2 int) {
	mr, _, moff := m.Size()
	mvec := m.elems.Slice()
	r1off, r2off := r1*moff, r2*moff
	for i := 0; i < mr; i++ {
		mvec[r1off+i], mvec[r2off+i] = mvec[r2off+i], mvec[r1off+i]
	}
}

// SetMesh copies part of the mesh, m into receiver, starting at (i, j)
func (n *mesh) SetMesh(m Mesher, i, j int) {
	mr, mc, _ := m.Size()
	for r := 0; r < mr; r++ {
		for c := 0; c < mc; c++ {
			n.SetAtNode(m.GetAtNode(r, c), i+r, j+c)
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
	mr, mc, _ := m.Size()
	nr, _, _ := n.Size()
	if mc != nr {
		panic("mesh: T: Receiver cols != Input rows.")
	}
	v := vec.Zeros(mr)
	for j := 0; j < mc; j++ {
		m.GetCol(v, j)
		n.SetRow(v, j)
	}
}

func (n *mesh) Triu(m Mesher, k int) {
	nr, nc, _ := m.Size()
	if k > 0 && k > nc || k < 0 && k < -nr {
		panic("mesh: Triu: requested diagonal out of range.")
	}

	for j := int(math.Max(0, float64(k))); j < nc; j++ {
		nr_lim := int(math.Min(float64(nr), float64(j-k)))
		for l := 0; l < nr_lim+1; l++ {
			n.SetAtNode(m.GetAtNode(l, j), l, j)
		}
	}
}

func (n *mesh) Tril(m Mesher, k int) {
	nr, nc, _ := m.Size()
	if k > 0 && k > nc || k < 0 && k < -nr {
		panic("mesh: Tril: requested diagonal out of range.")
	}

	for j := 0; j < int(math.Min(float64(nc), float64(nr+k))); j++ {
		nr_lim := int(math.Max(0, float64(j-k)))
		for l := nr_lim; l < nr; l++ {
			n.SetAtNode(m.GetAtNode(l, j), l, j)
		}
	}
}

func (m *mesh) IsSquare() bool {
	if r, c, _ := m.Size(); r != c {
		return false
	}
	return true
}

func (m *mesh) IsSymmetric() bool {
	n, _, _ := m.Size()
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if m.GetAtNode(i, j) != m.GetAtNode(j, i) {
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
		// vector to hold the columns
		v := vec.Zeros(m.r)
		// vector which holds the sums of the
		// columns
		sv := vec.Zeros(m.c)
		for j := 0; j < m.c; j++ {
			m.GetCol(v, j)
			sv.SetAt(j, v.AbsElemSum())
		}
		_, n := sv.MaxElem()
		return n
	case Two:
		sum := 0.
		for _, v := range m.elems.Slice() {
			sum += v * v
		}
		return math.Sqrt(sum)
	case Inf:
		v := vec.Zeros(m.c)
		// vector which holds the absolute sums of
		// the rows of mesh, m
		sv := vec.Zeros(m.r)
		for i := 0; i < m.r; i++ {
			m.GetRow(v, i)
			sv.SetAt(i, v.AbsElemSum())
		}
		_, n := sv.MaxElem()
		return n
	}
	return 0
}

// Scale multiplies a mesh with a scalar
func (m *mesh) Scale(a float64) { m.elems.Scale(a) }

// Inv computes the inverse of m into n
func (n *mesh) Inv(m Mesher) {
	eye := I(m.(*mesh).c)
	n.Solve(m, eye)
}

// Det computes the determinant of a square mesh, m
func (m mesh) Det() (det float64) {
	mc := m.c
	if mc == 3 {
		a := m.GetAtNode(0, 0)
		e := m.GetAtNode(1, 1)
		i := m.GetAtNode(2, 2)

		b := m.GetAtNode(0, 1)
		f := m.GetAtNode(1, 2)
		g := m.GetAtNode(2, 0)

		c := m.GetAtNode(0, 2)
		d := m.GetAtNode(1, 0)
		h := m.GetAtNode(2, 1)

		det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
		return
	} else if mc == 2 {
		a := m.GetAtNode(0, 0)
		d := m.GetAtNode(1, 1)

		b := m.GetAtNode(0, 1)
		c := m.GetAtNode(1, 0)

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

	ld := vec.Zeros(mc)
	ud := vec.Zeros(mc)

	L.GetDiag(ld)
	U.GetDiag(ud)

	detl, detu := 1., 1.
	uds := ud.Slice()
	for k, v := range ld.Slice() {
		detl *= v
		detu *= uds[k]
	}

	det = detl * detu
	nmuts(&det)
	return
}

// Sum computes the sum of meshes, m & n and puts
// result in receiver.
// m, n and o should have same dimensions.
// a*m + n = o
func (o *mesh) Sum(a float64, m, n Mesher) {
	_, mc, _ := m.Size()
	mv := vec.Zeros(mc)
	nv := vec.Zeros(mc)
	ov := vec.Zeros(mc)
	for j := 0; j < mc; j++ {
		m.GetCol(mv, j)
		n.GetCol(nv, j)
		ov.Sum(a, mv, nv)
		o.SetCol(ov, j)
	}
}

// Zero zeroes the mesh for reuse
func (m *mesh) Zero() {
	m.elems.Zero()
}

// IsEqual tells if two meshes are equal
func (n mesh) IsEqual(m Mesher) (iseq bool) {
	iseq = n.elems.IsEqual(m.Vec())
	return
}
