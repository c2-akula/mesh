package mesh

import "math"

// give the binary representation of integer s
func toBinary(s int) []int {
	_, t := math.Frexp(float64(s))
	var (
		bit = s
		bin = make([]int, t)
	)
	for k := t - 1; k >= 0; k-- {
		bin[k] = bit % 2
		bit /= 2
	}
	return bin
}

// Powm raises mesh, m to power s using
// binary powering and puts the result in
// receiver.
func (n *mesh) Powm(m Mesher, s int) {
	mr, mc := m.Size()
	if mr != mc {
		panic("Powm: Input mesh is not square.")
	}

	if s == 1 {
		n.Clone(m)
	} else if s == 0 {
		n.Clone(I(mc))
	}

	sbin := toBinary(s)
	k := len(sbin)
	i := k
	p := Gen(nil, mc, mc)
	p.Clone(m)
	b := Gen(nil, mc, mc)
	for sbin[i-1] == 0 {
		b.Mul(1, p, p)
		p.Clone(b)
		b.Zero()
		i--
	}
	n.Clone(p)
	for j := i - 2; j >= 0; j-- {
		b.Mul(1, p, p)
		p.Clone(b)
		if sbin[j] == 1 {
			b.Zero()
			b.Mul(1, n, p)
			n.Clone(b)
		}
		b.Zero()
	}
}

// Expm computes the exponential of a square mesh
// using the scaling and squaring method in conjunction
// with the taylor series.
func (n *mesh) Expm(m Mesher) {
	mr, mc := m.Size()
	nr, nc := n.Size()
	eye := I(mc)

	n.Clone(eye)
	if mr != mc || nr != nc {
		panic("Expm: Input mesh is not square.")
	}
	const N = 8 // scaling factor

	// Scaling part:
	// divide M by a factor of 2^N, so that
	// M/(2^N) is close enough to zero
	ms := make([]Mesher, 3, 3)
	for i := range ms {
		ms[i] = Gen(nil, mr, mc)
	}
	m_scaled := ms[0]
	m_scaled.Clone(m)
	m_scaled.Scale(math.Pow(2, -N))

	// Exponentiation part:
	// exp(M/(2^N)) is computed with high accuracy using
	// taylor series
	m_exp := ms[1]
	m_exp.Clone(m_scaled)
	fac := 1.
	const k = 10              // no. of factorial terms
	for i := 1.; i < k; i++ { // A + A2/2 + A3/3! + A4/4! + ...
		fac *= i
		m_exp.Powm(m_scaled, int(i))
		m_exp.Scale(1 / fac)
		n.Sum(1, 1, n, m_exp) // I + (A + A2/2 + A3/3! + A4/4! + ...)
	}

	// Squaring part:
	n.Powm(n, int(math.Pow(2, N)))
}
