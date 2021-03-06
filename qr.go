package mesh

import "math"

// QR decomposition of the given matrix using givens rotations
// The single givens rotations are stored within the matrix.
func QR(A Mesher) (Q, R Mesher) {
	m, n := A.Size()
	AC := Gen(nil, m, n)
	AC.Clone(A)

	// check in all columns except the n-th one for entries
	// to eliminate
	for j := 0; j < n-1; j++ {
		for i := j + 1; i < m; i++ {
			if AC.Get(i, j) != 0 {
				acjj := AC.Get(j, j)
				acij := AC.Get(i, j)
				r := math.Sqrt(acjj*acjj + acij*acij)

				if AC.Get(i, j) < 0 {
					r = -r
				}

				s := acij / r
				c := acjj / r

				// apply givens rotation
				for k := j; k < n; k++ {
					acjk := AC.Get(j, k)
					acik := AC.Get(i, k)
					AC.Set(c*acjk+s*acik, j, k)
					AC.Set(-s*acjk+c*acik, i, k)
				}

				// c & s can be stored in one matrix entry
				if c == 0 {
					AC.Set(1, i, j)
				} else if math.Abs(s) < math.Abs(c) {
					if c < 0 {
						AC.Set(-0.5*s, i, j)
					} else {
						AC.Set(0.5*s, i, j)
					}
				} else {
					AC.Set(2/c, i, j)
				}
			}
		}
	}

	// Q is an qxq matrix if m is the max of the no.rows & no.cols
	q := int(math.Max(float64(m), float64(n)))
	Q = I(q)

	for j := n - 1; j >= 0; j-- {
		for i := m - 1; i > j; i-- {
			// Get c & s stored in the i-th row, j-th col
			acij := AC.Get(i, j)

			c, s := 0., 0.
			if acij == 0 {
				c, s = 0., 1.
			} else if math.Abs(acij) < 1 {
				s = 2 * math.Abs(acij)
				c = math.Sqrt(1 - s*s)
				if acij < 0 {
					c = -c
				}
			} else {
				c = 2. / acij
				s = math.Sqrt(1 - c*c)
			}

			for k := 0; k < n; k++ {
				acjk := Q.Get(j, k)
				acik := Q.Get(i, k)

				Q.Set(c*acjk-s*acik, j, k)
				Q.Set(s*acjk+c*acik, i, k)
			}
		}
	}

	// R is upper triangular matrix
	R = Gen(nil, n, n)
	R.Clone(AC)

	for i := 0; i < m; i++ {
		for j := 0; j < i; j++ {
			R.Set(0, i, j)
		}
	}

	return
}
