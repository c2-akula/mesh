package mesh

import "math"

// The implementation of the eigenvalue decomposition
// algorithm has been adapted from JAMA.
// http://www.iaa.ncku.edu.tw/~dychiang/lab/program/mohr3d/source/Jama%5CEigenvalueDecomposition.html
type eigen struct {
	n int // square matrix size
	// storage of eigenvectors
	v Mesher
	// storage of non-symmetric
	// hessenberg form
	h Mesher

	// storage for real & complex eigenvalues
	d, e []float64
}

type Eigener interface {
	// EigenVector Matrix
	V() Mesher
	// Block Diagonal Eigenvalue matrix
	D() Mesher
	// Real parts of the eigenvalues
	RealEigVal() []float64
	// Complex parts of the eigenvalues
	CompEigVal() []float64
}

// Returns the eigenvector matrix
func (e eigen) V() Mesher {
	return e.v
}

// D returns the block diagonal eigenvalue matrix
func (e eigen) D() Mesher {
	n := e.n
	x := Gen(nil, n, n)
	for i := 0; i < n; i++ {
		x.Set(e.d[i], i, i)
		if e.e[i] > 0 {
			x.Set(e.e[i], i, i+1)
		} else if e.e[i] < 0 {
			x.Set(e.e[i], i, i-1)
		}
	}
	return x
}

// RealEigVal returns the real parts of the eigenvalues
func (e eigen) RealEigVal() []float64 {
	return e.d
}

// CompEigVal returns the complex parts of the eigenvalues
func (e eigen) CompEigVal() []float64 {
	return e.e
}

// tred2 - symmetric householder reduction of v to tridiagonal form
func tred2(d, e []float64, v Mesher) {
	n, _ := v.Size()
	for j := 0; j < n; j++ {
		d[j] = v.Get(n-1, j)
	}
	// householder reduction to tridiagonal form
	for i := n - 1; i > 0; i-- {
		// scale to avoid under/overflow
		scale := 0.0
		h := 0.0
		for k := 0; k < i; k++ {
			scale += math.Abs(d[k])
		}
		if scale == 0.0 {
			e[i] = d[i-1]
			for j := 0; j < i; j++ {
				d[j] = v.Get(i-1, j)
				v.Set(0.0, i, j)
				v.Set(0.0, j, i)
			}
		} else {
			// generate householder vector
			for k := 0; k < i; k++ {
				d[k] /= scale
				h += d[k] * d[k]
			}
			f := d[i-1]
			g := math.Sqrt(h)
			if f > 0 {
				g = -g
			}
			e[i] = scale * g
			h -= f * g
			d[i-1] = f - g
			for j := 0; j < i; j++ {
				e[j] = 0.0
			}

			// apply similarity transformation to remaining columns
			for j := 0; j < i; j++ {
				f = d[j]
				v.Set(f, j, i)
				g = e[j] + v.Get(j, j)*f
				for k := j + 1; k <= i-1; k++ {
					g += v.Get(k, j) * d[k]
					e[k] += v.Get(k, j) * f
				}
				e[j] = g
			}
			f = 0.0
			for j := 0; j < i; j++ {
				e[j] /= h
				f += e[j] * d[j]
			}
			hh := f / (h + h)
			for j := 0; j < i; j++ {
				e[j] -= hh * d[j]
			}
			for j := 0; j < i; j++ {
				f = d[j]
				g = e[j]
				for k := j; k <= i-1; k++ {
					v.Set(v.Get(k, j)-(f*e[k]+g*d[k]), k, j)
				}
				d[j] = v.Get(i-1, j)
				v.Set(0.0, i, j)
			}
		}
		d[i] = h
	}
	// accumulate transformations
	for i := 0; i < n-1; i++ {
		v.Set(v.Get(i, i), n-1, i)
		v.Set(1.0, i, i)
		h := d[i+1]
		if h != 0.0 {
			for k := 0; k <= i; k++ {
				d[k] = v.Get(k, i+1) / h
			}
			for j := 0; j <= i; j++ {
				g := 0.0
				for k := 0; k <= i; k++ {
					g += v.Get(k, i+1) * v.Get(k, j)
				}
				for k := 0; k <= i; k++ {
					v.Set(v.Get(k, j)-g*d[k], k, j)
				}
			}
		}
		for k := 0; k <= i; k++ {
			v.Set(0.0, k, i+1)
		}
	}
	for j := 0; j < n; j++ {
		d[j] = v.Get(n-1, j)
		v.Set(0.0, n-1, j)
	}
	v.Set(1.0, n-1, n-1)
	e[0] = 0.0
}

// tql2, symmetric tridiagonal QL algorithm
func tql2(d, e []float64, v Mesher) {
	n, _ := v.Size()

	for i := 1; i < n; i++ {
		e[i-1] = e[i]
	}
	e[n-1] = 0.0

	f := 0.0
	tst1 := 0.0
	eps := Eps
	for l := 0; l < n; l++ {
		// find small subdiagonal element
		tst1 = math.Max(tst1, math.Abs(d[l])+math.Abs(e[l]))
		m := l
		for m < n {
			if math.Abs(e[m]) <= eps*tst1 {
				break
			}
			m++
		}

		// if m == 1, d[l] is an eigenvalue
		// otherwise, iterate
		if m > l {
			for iter := 0; ; iter++ {

				// compute implicit shift
				g := d[l]
				p := (d[l+1] - g) / (2.0 * e[l])
				r := math.Hypot(p, 1.0)
				if p < 0 {
					r = -r
				}
				d[l] = e[l] / (p + r)
				d[l+1] = e[l] * (p + r)
				dl1 := d[l+1]
				h := g - d[l]
				for i := l + 2; i < n; i++ {
					d[i] -= h
				}
				f += h

				// implicit QL transformation
				p = d[m]
				c := 1.0
				c2, c3 := c, c
				el1 := e[l+1]
				s := 0.0
				s2 := 0.0
				for i := m - 1; i >= l; i-- {
					c3 = c2
					c2 = c
					s2 = s
					g = c * e[i]
					h = c * p
					r = math.Hypot(p, e[i])
					e[i+1] = s * r
					s = e[i] / r
					c = p / r
					p = c*d[i] - s*g
					d[i+1] = h + s*(c*g+s*d[i])

					// accumulate transformation
					for k := 0; k < n; k++ {
						h = v.Get(k, i+1)
						v.Set(s*v.Get(k, i)+c*h, k, i+1)
						v.Set(c*v.Get(k, i)-s*h, k, i)
					}
				}
				p = -s * s2 * c3 * el1 * e[l] / dl1
				e[l] = s * p
				d[l] = c * p

				// check for convergence
				if math.Abs(e[l]) <= eps*tst1 {
					break
				}
			}
		}
		d[l] += f
		e[l] = 0.0
	}
	// sort eigenvalues and corresponding vectors
	for i := 0; i < n-1; i++ {
		k := i
		p := d[i]
		for j := i + 1; j < n; j++ {
			if d[j] < p {
				k = j
				p = d[j]
			}
		}
		if k != i {
			d[k] = d[i]
			d[i] = p
			for j := 0; j < n; j++ {
				p = v.Get(j, i)
				v.Set(v.Get(j, k), j, i)
				v.Set(p, j, k)
			}
		}
	}
}

// complex scalar division
func cdiv(xr, xi, yr, yi float64) (cdivr, cdivi float64) {
	c1 := complex(xr, xi)
	c2 := complex(yr, yi)
	div := c1 / c2
	cdivr = real(div)
	cdivi = imag(div)
	return
}

// hess reduces non-symmetric matrix to hessenberg form.
func hess(H, V Mesher) {
	n, _ := H.Size()
	low, high := 0, n-1

	// working storage for nonsymmetric algorithm
	ort := make([]float64, n)
	for m := low + 1; m <= high-1; m++ {
		// scale column
		scale := 0.0
		for i := m; i <= high; i++ {
			scale += math.Abs(H.Get(i, m-1))
		}

		if scale != 0.0 {
			// compute householder transformation
			h := 0.0
			for i := high; i >= m; i-- {
				ort[i] = H.Get(i, m-1) / scale
				h += ort[i] * ort[i]
			}
			g := math.Sqrt(h)
			if ort[m] > 0 {
				g = -g
			}
			h -= ort[m] * g
			ort[m] -= g

			// apply householder similarity transformation
			// H = (I - u*u'/h)*H*(I-u*u'/h)

			for j := m; j < n; j++ {
				f := 0.0
				for i := high; i >= m; i-- {
					f += ort[i] * H.Get(i, j)
				}
				f /= h
				for i := m; i <= high; i++ {
					H.Set(H.Get(i, j)-f*ort[i], i, j)
				}
			}

			for i := 0; i <= high; i++ {
				f := 0.0
				for j := high; j >= m; j-- {
					f += ort[j] * H.Get(i, j)
				}
				f /= h
				for j := m; j <= high; j++ {
					H.Set(H.Get(i, j)-f*ort[j], i, j)
				}
			}
			ort[m] = scale * ort[m]
			H.Set(scale*g, m, m-1)
		}
	}
	// accumulate transformations
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				V.Set(1.0, i, j)
			} else {
				V.Set(0.0, i, j)
			}
		}
	}

	for m := high - 1; m >= low+1; m-- {
		if H.Get(m, m-1) != 0.0 {
			for i := m + 1; i <= high; i++ {
				ort[i] = H.Get(i, m-1)
			}
			for j := m; j <= high; j++ {
				g := 0.0
				for i := m; i <= high; i++ {
					g += ort[i] * V.Get(i, j)
				}
				// double division avoids possible underflow
				g = (g / ort[m]) / H.Get(m, m-1)
				for i := m; i <= high; i++ {
					V.Set(V.Get(i, j)+g*ort[i], i, j)
				}
			}
		}
	}
}

// hqr2 reduces from hessenberg to real schur form
func hqr2(d, e []float64, H, V Mesher) {
	// initialize
	nn, _ := H.Size()
	n := nn - 1
	low, high := 0, nn-1
	eps := Eps
	var (
		exshift, p, q, r, s, z, t, w, x, y float64
	)

	// store roots isolated by balanc and compute matrix norm
	norm := 0.0
	for i := 0; i < nn; i++ {
		if i < low || i > high {
			d[i] = H.Get(i, i)
			e[i] = 0.0
		}
		for j := int(math.Max(float64(i-1), 0)); j < nn; j++ {
			norm += math.Abs(H.Get(i, j))
		}
	}

	// Outer loop over eigenvalue index
	iter := 0
	for n >= low {
		// look for single small subdiagonal element
		l := n
		for l > low {
			s = math.Abs(H.Get(l-1, l-1)) + math.Abs(H.Get(l, l))
			if s == 0.0 {
				s = norm
			}
			if math.Abs(H.Get(l, l-1)) < eps*s {
				break
			}
			l--
		}

		// check for convergence
		// one root found
		if l == n {
			H.Set(H.Get(n, n)+exshift, n, n)
			d[n] = H.Get(n, n)
			e[n] = 0.0
			n--
			iter = 0

			// two roots found
		} else if l == n-1 {
			w = H.Get(n, n-1) * H.Get(n-1, n)
			p = (H.Get(n-1, n-1) - H.Get(n, n)) / 2.0
			q = p*p + w
			z = math.Sqrt(math.Abs(q))
			H.Set(H.Get(n, n)+exshift, n, n)
			H.Set(H.Get(n-1, n-1)+exshift, n-1, n-1)
			x = H.Get(n, n)

			// real pair
			if q >= 0 {
				if p >= 0 {
					z = p + z
				} else {
					z = p - z
				}
				d[n-1] = x + z
				d[n] = d[n-1]
				if z != 0.0 {
					d[n] = x - w/z
				}
				e[n-1] = 0.0
				e[n] = 0.0
				x = H.Get(n, n-1)
				s = math.Abs(x) + math.Abs(z)
				p = x / s
				q = z / s
				r = math.Sqrt(p*p + q*q)
				p /= r
				q /= r

				// row modification
				for j := n - 1; j < nn; j++ {
					z = H.Get(n-1, j)
					H.Set(q*z+p*H.Get(n, j), n-1, j)
					H.Set(q*H.Get(n, j)-p*z, n, j)
				}
				// column modification
				for i := 0; i <= n; i++ {
					z = H.Get(i, n-1)
					H.Set(q*z+p*H.Get(i, n), i, n-1)
					H.Set(q*H.Get(i, n)-p*z, i, n)
				}

				// accumulate transformations
				for i := low; i <= high; i++ {
					z = V.Get(i, n-1)
					V.Set(q*z+p*V.Get(i, n), i, n-1)
					V.Set(q*V.Get(i, n)-p*z, i, n)
				}

				// complex pair
			} else {
				d[n-1] = x + p
				d[n] = x + p
				e[n-1] = z
				e[n] = -z
			}
			n = n - 2
			iter = 0

			// no convergence yet
		} else {
			// form shift
			x = H.Get(n, n)
			y, w = 0.0, 0.0
			if l < n {
				y = H.Get(n-1, n-1)
				w = H.Get(n, n-1) * H.Get(n-1, n)
			}

			// wilkinson's original ad hoc shift
			if iter == 10 {
				exshift += x
				for i := low; i <= n; i++ {
					H.Set(H.Get(i, i)-x, i, i)
				}
				s = math.Abs(H.Get(n, n-1)) + math.Abs(H.Get(n-1, n-2))
				x = 0.75 * s
				y = x
				w = -0.4375 * s * s
			}

			// MATLAB's new ad hoc shift
			if iter == 30 {
				s = (y - x) / 2.0
				s = s*s + w
				if s > 0 {
					s = math.Sqrt(s)
					if y < x {
						s = -s
					}
					s = x - w/((y-x)/2.0+s)
					for i := low; i <= n; i++ {
						H.Set(H.Get(i, i)-s, i, i)
					}
					exshift += s
					x = 0.964
					y, w = x, x
				}
			}
			iter++

			// look for two consecutive small subdiagonal elements
			m := n - 2
			for m >= l {
				z = H.Get(m, m)
				r = x - z
				s = y - z
				p = (r*s-w)/H.Get(m+1, m) + H.Get(m, m+1)
				q = H.Get(m+1, m+1) - z - r - s
				r = H.Get(m+2, m+1)
				s = math.Abs(p) + math.Abs(q) + math.Abs(r)
				p /= s
				q /= s
				r /= s
				if m == l {
					break
				}

				// if H[m,m-1]*(q+r) < eps*(p*(H[m-1,m-1]) + z + H[m+1,m+1])
				if math.Abs(H.Get(m, m-1))*(math.Abs(q)+math.Abs(r)) <
					eps*(math.Abs(p)*(math.Abs(H.Get(m-1, m-1))+math.Abs(z)+
						math.Abs(H.Get(m+1, m+1)))) {
					break
				}
				m--
			}

			for i := m + 2; i <= n; i++ {
				H.Set(0.0, i, i-2)
				if i > m+2 {
					H.Set(0.0, i, i-3)
				}
			}

			// Double QR step involving rows l:n and columns m:n
			for k := m; k <= n-1; k++ {
				notlast := (k != n-1)
				if k != m {
					p = H.Get(k, k-1)
					q = H.Get(k+1, k-1)
					if notlast {
						r = H.Get(k+2, k-1)
					} else {
						r = 0.0
					}
					x = math.Abs(p) + math.Abs(q) + math.Abs(r)
					if x != 0.0 {
						p /= x
						q /= x
						r /= x
					}
				}
				if x == 0 {
					break
				}
				s = math.Sqrt(p*p + q*q + r*r)
				if p < 0 {
					s = -s
				}
				if s != 0 {
					if k != m {
						H.Set(-s*x, k, k-1)
					} else if l != m {
						H.Set(-H.Get(k, k-1), k, k-1)
					}
					p += s
					x = p / s
					y = q / s
					z = r / s
					q /= p
					r /= p

					// row modification
					for j := k; j < nn; j++ {
						p = H.Get(k, j) + q*H.Get(k+1, j)
						if notlast {
							p = p + r*H.Get(k+2, j)
							H.Set(H.Get(k+2, j)-p*z, k+2, j)
						}
						H.Set(H.Get(k, j)-p*x, k, j)
						H.Set(H.Get(k+1, j)-p*y, k+1, j)
					}

					// column modification
					for i := 0; i <= int(math.Min(float64(n), float64(k+3))); i++ {
						p = x*H.Get(i, k) + y*H.Get(i, k+1)
						if notlast {
							p = p + z*H.Get(i, k+2)
							H.Set(H.Get(i, k+2)-p*r, i, k+2)
						}
						H.Set(H.Get(i, k)-p, i, k)
						H.Set(H.Get(i, k+1)-p*q, i, k+1)
					}

					// accumulate transformations
					for i := low; i <= high; i++ {
						p = x*V.Get(i, k) + y*V.Get(i, k+1)
						if notlast {
							p = p + z*V.Get(i, k+2)
							V.Set(V.Get(i, k+2)-p*r, i, k+2)
						}
						V.Set(V.Get(i, k)-p, i, k)
						V.Set(V.Get(i, k+1)-p*q, i, k+1)
					}
				} // s != 0
			} // k loop
		} // check convergence
	} // while (n >= low)

	// backsubstitute to find vectors of upper triangular form
	if norm == 0.0 {
		return
	}

	for n = nn - 1; n >= 0; n-- {
		p = d[n]
		q = e[n]
		// real vector
		if q == 0 {
			l := n
			H.Set(1.0, n, n)
			for i := n - 1; i >= 0; i-- {
				w = H.Get(i, i) - p
				r = 0.0
				for j := l; j <= n; j++ {
					r = r + H.Get(i, j)*H.Get(j, n)
				}
				if e[i] < 0.0 {
					z = w
					s = r
				} else {
					l = i
					if e[i] == 0.0 {
						if w != 0.0 {
							H.Set(-r/w, i, n)
						} else {
							H.Set(-r/(eps*norm), i, n)
						}
					} else {
						// solve real equations
						x = H.Get(i, i+1)
						y = H.Get(i+1, i)
						q = (d[i]-p)*(d[i]-p) + e[i]*e[i]
						t = (x*s - z*r) / q
						H.Set(t, i, n)
						if math.Abs(x) > math.Abs(z) {
							H.Set((-r-w*t)/x, i+1, n)
						} else {
							H.Set((-s-y*t)/z, i+1, n)
						}
					}

					// overflow control
					t = math.Abs(H.Get(i, n))
					if (eps*t)*t > 1 {
						for j := i; j <= n; j++ {
							H.Set(H.Get(j, n)/t, j, n)
						}
					}
				}
			}
			// complex vector
		} else if q < 0 {
			l := n - 1

			// last vector component imaginary so matrix is
			// triangular
			if math.Abs(H.Get(n, n-1)) > math.Abs(H.Get(n-1, n)) {
				H.Set(q/H.Get(n, n-1), n-1, n-1)
				H.Set(-(H.Get(n, n)-p)/H.Get(n, n-1), n-1, n)
			} else {
				cdivr, cdivi := cdiv(0.0, -H.Get(n-1, n), H.Get(n-1, n-1)-p, q)
				H.Set(cdivr, n-1, n-1)
				H.Set(cdivi, n-1, n)
			}
			H.Set(0.0, n, n-1)
			H.Set(1.0, n, n)
			for i := n - 2; i >= 0; i-- {
				var ra, sa, vr, vi float64
				for j := l; j <= n; j++ {
					ra = ra + H.Get(i, j)*H.Get(j, n-1)
					sa = sa + H.Get(i, j)*H.Get(j, n)
				}
				w = H.Get(i, i) - p
				if e[i] < 0.0 {
					z = w
					r = ra
					s = sa
				} else {
					l = i
					if e[i] == 0 {
						cdivr, cdivi := cdiv(-ra, -sa, w, q)
						H.Set(cdivr, i, n-1)
						H.Set(cdivi, i, n)
					} else {
						// solve complex equations
						x = H.Get(i, i+1)
						y = H.Get(i+1, i)
						vr = (d[i]-p)*(d[i]-p) + e[i]*e[i] - q*q
						vi = (d[i] - p) * 2.0 * q
						if vr == 0.0 && vi == 0.0 {
							vr = eps * norm * (math.Abs(w) + math.Abs(q) +
								math.Abs(x) + math.Abs(y) + math.Abs(z))
						}
						cdivr, cdivi := cdiv(x*r-z*ra+q*sa, x*s-z*sa-q*ra, vr, vi)
						H.Set(cdivr, i, n-1)
						H.Set(cdivi, i, n)
						if math.Abs(x) > math.Abs(z)+math.Abs(q) {
							H.Set((-ra-w*H.Get(i, n-1)+q*H.Get(i, n))/x, i+1, n-1)
							H.Set((-sa-w*H.Get(i, n)-q*H.Get(i, n-1))/x, i+1, n)
						} else {
							cdivr, cdivi := cdiv(-r-y*H.Get(i, n-1), -s-y*H.Get(i, n), z, q)
							H.Set(cdivr, i+1, n-1)
							H.Set(cdivi, i+1, n)
						}
					}
					// overflow control
					t = math.Max(math.Abs(H.Get(i, n-1)), math.Abs(H.Get(i, n)))
					if (eps*t)*t > 1.0 {
						for j := i; j <= n; j++ {
							H.Set(H.Get(j, n-1)/t, j, n-1)
							H.Set(H.Get(j, n)/t, j, n)
						}
					}
				}
			}
		}
	}

	// vectors of isolated roots
	for i := 0; i < nn; i++ {
		if i < low || i > high {
			for j := i; j < nn; j++ {
				V.Set(H.Get(i, j), i, j)
			}
		}
	}

	// back transformation to get eigenvectors of original matrix
	for j := nn - 1; j >= low; j-- {
		for i := low; i <= high; i++ {
			z = 0.0
			for k := low; k <= int(math.Min(float64(j), float64(high))); k++ {
				z = z + V.Get(i, k)*H.Get(k, j)
			}
			V.Set(z, i, j)
		}
	}
}

func Eig(a Mesher) Eigener {
	eig := new(eigen)
	if !a.IsSquare() {
		panic("Eig: Input matrix should be square.")
	}

	n, _ := a.Size()
	eig.n = n
	eig.v = Gen(nil, n, n)
	eig.d = make([]float64, n) // real parts of eigenvalues
	eig.e = make([]float64, n) // imaginary parts of eigenvalues

	if a.IsSymmetric() {
		eig.v.Clone(a)

		// tridiagonalize
		tred2(eig.d, eig.e, eig.v)

		// diagonalize
		tql2(eig.d, eig.e, eig.v)
	} else {
		eig.h = Gen(nil, n, n)
		eig.h.Clone(a)

		// reduce to hessenberg form
		hess(eig.h, eig.v)

		// reduce hessenberg form to real schur form
		hqr2(eig.d, eig.e, eig.h, eig.v)
	}
	return eig
}
