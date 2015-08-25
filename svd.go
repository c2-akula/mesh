package mesh

import "math"

type svd struct {
	u, v, s []float64
	m, n    int
}

type SVDer interface {
	U() (U Mesher)
	V() (V Mesher)
	S() (S Mesher)
	SingularVals() (sv []float64)
	Norm2() float64
	Cond() float64
	Rank() int
}

// SVD computes the singular value decomposition of an
// mxn matrix A.
// SVD results in following decomposition,
// if m >= n
// 	U - mxn
// 	S - nxn
// 	V - nxn
// if m < n
// 	U - mxm
// 	S - mxn
// 	V - nxn
// Singualar Values are ordered so that
// sigma[0] >= sigma[1] >= ... >= sigma[n-1]
// Singular value decomposition always exists
// and will therefore never fail.
// The matrix condition number & effective numerical
// rank can be computed from this decomposition
// http://www.iaa.ncku.edu.tw/~dychiang/lab/program/mohr3d/source/Jama/SingularValueDecomposition.html
func SVD(a Mesher) SVDer {
	var sqa Mesher
	svd := new(svd)
	m, n := a.Size()
	svd.m, svd.n = m, n
	if m < n { // 6x7
		// we will make it square by appending
		// a zero column
		sqa = Gen(nil, n, n) // 7x7
		m = n
		// svd.m = m
	} else if m >= n { // 7x6
		sqa = Gen(nil, m, m) // 7x7
		n = m
		// svd.n = n
	}
	sqa.Clone(a)
	svd.u = make([]float64, m*n) // 7x7
	nu := n
	svd.s = make([]float64, n)   // 7x7
	svd.v = make([]float64, n*n) // 7x7

	e := make([]float64, n)    // 7
	work := make([]float64, m) // 7
	wantu, wantv := true, true

	// reduce A to bidiagonal form, storing the diagonal
	// elements in s and the super-diagonal elements in e
	nct := int(math.Min(float64(m-1), float64(n)))              // 6 = min(6, 7)
	nrt := int(math.Max(0, math.Min(float64(n-2), float64(m)))) // 5 = max(0, min(5, 7))

	for k := 0; k < int(math.Max(float64(nct), float64(nrt))); k++ {
		if k < nct {

			// compute the transformation for the kth column and place
			// the kth diagonal in s[k]
			// compute 2-norm of k-th column without under/overflow
			svd.s[k] = 0
			for i := k; i < m; i++ {
				svd.s[k] = math.Hypot(svd.s[k], sqa.Get(i, k))
			}
			if svd.s[k] != 0.0 {
				if sqa.Get(k, k) < 0.0 {
					svd.s[k] = -svd.s[k]
				}
				for i := k; i < m; i++ {
					sqa.Set(sqa.Get(i, k)/svd.s[k], i, k)
				}
				sqa.Set(sqa.Get(k, k)+1.0, k, k)
			}
			svd.s[k] = -svd.s[k]
		}
		for j := k + 1; j < n; j++ {
			if k < nct && svd.s[k] != 0 {
				// apply the transformation
				t := 0.0
				for i := k; i < m; i++ {
					t += sqa.Get(i, k) * sqa.Get(i, j)
				}
				t /= -sqa.Get(k, k)
				for i := k; i < m; i++ {
					sqa.Set(sqa.Get(i, j)+t*sqa.Get(i, k), i, j)
				}
			}

			// place the kth row of A into e for the subsequent calculation of
			// the row transformation
			e[j] = sqa.Get(k, j)
		}

		if wantu && (k < nct) {
			// place the transformation in U for subsequent back multiplication
			for i := k; i < m; i++ {
				svd.u[i*n+k] = sqa.Get(i, k)
			}
		}

		if k < nrt {
			// compute the kth row transformation and place the
			// kth superdiagonal in e[k]
			// compute the 2-norm without under/overflow
			e[k] = 0
			for i := k + 1; i < n; i++ {
				e[k] = math.Hypot(e[k], e[i])
			}
			if e[k] != 0.0 {
				if e[k+1] < 0.0 {
					e[k] = -e[k]
				}

				for i := k + 1; i < n; i++ {
					e[i] /= e[k]
				}
				e[k+1] += 1.0
			}
			e[k] = -e[k]
			if (k+1 < m) && (e[k] != 0.0) {
				// apply the transformation
				for i := k + 1; i < m; i++ {
					work[i] = 0.0
				}
				for j := k + 1; j < n; j++ {
					for i := k + 1; i < m; i++ {
						work[i] += e[j] * sqa.Get(i, j)
					}
				}
				for j := k + 1; j < n; j++ {
					t := -e[j] / e[k+1]
					for i := k + 1; i < m; i++ {
						sqa.Set(sqa.Get(i, j)+t*work[i], i, j)
					}
				}
			}
			if wantv {
				// place the transformation in V for subsequent back
				// multiplication
				for i := k + 1; i < n; i++ {
					svd.v[i*n+k] = e[i]
				}
			}
		}
	}

	// set up the final bidiagonal matrix of order p
	p := int(math.Min(float64(n), float64(m+1)))
	if nct < n {
		svd.s[nct] = sqa.Get(nct, nct)
	}
	if m < p {
		svd.s[p-1] = 0.0
	}
	if nrt+1 < p {
		e[nrt] = sqa.Get(nrt, p-1)
	}
	e[p-1] = 0.0

	// if required generate U
	if wantu {
		for j := nct; j < nu; j++ {
			for i := 0; i < m; i++ {
				svd.u[i*n+j] = 0.0
			}
			svd.u[j*n+j] = 1.0
		}

		for k := nct - 1; k >= 0; k-- {
			if svd.s[k] != 0 {
				for j := k + 1; j < nu; j++ {
					t := 0.0
					for i := k; i < m; i++ {
						t += svd.u[i*n+k] * svd.u[i*n+j]
					}
					t /= -svd.u[k*n+k]
					for i := k; i < m; i++ {
						svd.u[i*n+j] += t * svd.u[i*n+k]
					}
				}
				for i := k; i < m; i++ {
					svd.u[i*n+k] = -svd.u[i*n+k]
				}
				svd.u[k*n+k] += 1.0
				for i := 0; i < k-1; i++ {
					svd.u[i*n+k] = 0.0
				}
			} else {
				for i := 0; i < m; i++ {
					svd.u[i*n+k] = 0.0
				}
				svd.u[k*n+k] = 1.0
			}
		}
	}

	// if required, generate v
	if wantv {
		for k := n - 1; k >= 0; k-- {
			if k < nrt && e[k] != 0.0 {
				for j := k + 1; j < nu; j++ {
					t := 0.0
					for i := k + 1; i < n; i++ {
						t += svd.v[i*n+k] * svd.v[i*n+j]
					}
					t /= -svd.v[(k+1)*n+k]
					for i := k + 1; i < n; i++ {
						svd.v[i*n+j] += t * svd.v[i*n+k]
					}
				}
			}
			for i := 0; i < n; i++ {
				svd.v[i*n+k] = 0.0
			}
			svd.v[k*n+k] = 1.0
		}
	}

	// main iteration loop for the singualr values
	pp := p - 1
	iter := 0
	eps := math.Pow(2.0, -52.0)
	for p > 0 {
		k, kase := 0, 0

		// here is where a test for too many iterations would go
		//
		// this section of the program inspects for negligible
		// elements in the s and e arrays. On completion the
		// variables kase and k are set as follows,
		// kase = 1 if s(p) && e[k-1] are negligible && k<p
		// kase = 2 if s(k) is negligible && k < p
		// kase = 3     if e[k-1] is negligible, k<p, and
		//              s(k), ..., s(p) are not negligible (qr step).
		// kase = 4     if e(p-1) is negligible (convergence).

		for k = p - 2; k >= -1; k-- {
			if k == -1 {
				break
			}
			if math.Abs(e[k]) <= eps*(math.Abs(svd.s[k])+math.Abs(svd.s[k+1])) {
				e[k] = 0.0
				break
			}
		}
		if k == p-2 {
			kase = 4
		} else {
			ks := 0
			for ks = p - 1; ks >= k; ks-- {
				if ks == k {
					break
				}
				t := 0.0
				if ks != p {
					t = math.Abs(e[ks])
				}
				if ks != k+1 {
					t += math.Abs(e[ks-1])
				} else {
					t += 0.0
				}
				if math.Abs(svd.s[ks]) <= eps*t {
					svd.s[ks] = 0.0
					break
				}
			}
			if ks == k {
				kase = 3
			} else if ks == p-1 {
				kase = 1
			} else {
				kase = 2
				k = ks
			}
		}
		k++

		// perform the task indicated by kase
		switch kase {
		case 1:
			{
				// deflate negligible s[p]
				f := e[p-2]
				e[p-2] = 0.0
				for j := p - 2; j >= k; j-- {
					t := math.Hypot(svd.s[j], f)
					cs := svd.s[j] / t
					sn := f / t
					svd.s[j] = t
					if j != k {
						f = -sn * e[j-1]
						e[j-1] *= cs
					}
					if wantv {
						for i := 0; i < n; i++ {
							t = cs*svd.v[i*n+j] + sn*svd.v[i*n+(p-1)]
							svd.v[i*n+(p-1)] = cs*svd.v[i*n+(p-1)] - sn*svd.v[i*n+j]
							svd.v[i*n+j] = t
						}
					}
				}
			}
		case 2:
			{
				// split at negligible s[k]
				f := e[k-1]
				e[k-1] = 0.0
				for j := k; j < p; j++ {
					t := math.Hypot(svd.s[j], f)
					cs := svd.s[j] / t
					sn := f / t
					svd.s[j] = t
					f = -sn * e[j]
					e[j] *= cs
					if wantu {
						for i := 0; i < m; i++ {
							t = cs*svd.u[i*n+j] + sn*svd.u[i*n+(k-1)]
							svd.u[i*n+(k-1)] = -sn*svd.u[i*n+j] + cs*svd.u[i*n+(k-1)]
							svd.u[i*n+j] = t
						}
					}
				}
			}
		case 3:
			{
				// perform one qr step
				//
				// calculate the shift
				scale := math.Max(math.Max(math.Max(math.Max(
					math.Abs(svd.s[p-1]), math.Abs(svd.s[p-2])), math.Abs(e[p-2])),
					math.Abs(svd.s[k])), math.Abs(e[k]))
				sp := svd.s[p-1] / scale
				spm1 := svd.s[p-2] / scale
				epm1 := e[p-2] / scale
				sk := svd.s[k] / scale
				ek := e[k] / scale
				b := ((spm1+sp)*(spm1-sp) + epm1*epm1) / 2.0
				c := (sp * epm1) * (sp * epm1)
				shift := 0.0
				if b != 0.0 || c != 0.0 {
					shift = math.Sqrt(b*b + c)
					if b < 0.0 {
						shift = -shift
					}
					shift = c / (b + shift)
				}
				f := (sk+sp)*(sk-sp) + shift
				g := sk * ek

				// chase zeros
				for j := k; j < p-1; j++ {
					t := math.Hypot(f, g)
					cs := f / t
					sn := g / t
					if j != k {
						e[j-1] = t
					}
					f = cs*svd.s[j] + sn*e[j]
					e[j] = cs*e[j] - sn*svd.s[j]
					g = sn * svd.s[j+1]
					svd.s[j+1] *= cs
					if wantv {
						for i := 0; i < n; i++ {
							t = cs*svd.v[i*n+j] + sn*svd.v[i*n+(j+1)]
							svd.v[i*n+(j+1)] = -sn*svd.v[i*n+j] + cs*svd.v[i*n+(j+1)]
							svd.v[i*n+j] = t
						}
					}
					t = math.Hypot(f, g)
					cs = f / t
					sn = g / t
					svd.s[j] = t
					f = cs*e[j] + sn*svd.s[j+1]
					svd.s[j+1] = -sn*e[j] + cs*svd.s[j+1]
					g = sn * e[j+1]
					e[j+1] = cs * e[j+1]
					if wantu && j < m-1 {
						for i := 0; i < m; i++ {
							t = cs*svd.u[i*n+j] + sn*svd.u[i*n+(j+1)]
							svd.u[i*n+(j+1)] = -sn*svd.u[i*n+j] + cs*svd.u[i*n+(j+1)]
							svd.u[i*n+j] = t
						}
					}
				}
				e[p-2] = f
				iter++
			}
		case 4:
			{
				// TODO implementation
				//
				// make the singular values positive
				if svd.s[k] <= 0 {
					if svd.s[k] < 0 {
						svd.s[k] = -svd.s[k]
					} else {
						svd.s[k] = 0
					}
					if wantv {
						for i := 0; i <= pp; i++ {
							svd.v[i*n+k] = -svd.v[i*n+k]
						}
					}
				}

				// order the singular values
				for k < pp {
					if svd.s[k] >= svd.s[k+1] {
						break
					}
					t := svd.s[k]
					svd.s[k] = svd.s[k+1]
					svd.s[k+1] = t
					if wantv && k < n-1 {
						for i := 0; i < n; i++ {
							t = svd.v[i*n+(k+1)]
							svd.v[i*n+(k+1)] = svd.v[i*n+k]
							svd.v[i*n+k] = t
						}
					}
					if wantu && k < m-1 {
						for i := 0; i < m; i++ {
							t = svd.u[i*n+(k+1)]
							svd.u[i*n+(k+1)] = svd.u[i*n+k]
							svd.u[i*n+k] = t
						}
					}
					k++
				}
				iter = 0
				p--
			}
		}
	}
	return svd
}

// U returns the left singular values
func (s svd) U() (U Mesher) {
	if s.m < s.n {
		U = Gen(nil, s.m, s.m)
		for i := 0; i < s.m; i++ {
			for j := 0; j < s.m; j++ {
				U.Set(s.u[i*s.n+j], i, j)
			}
		}
	} else {
		U = Gen(nil, s.m, s.n)
		copy(U.Slice(), s.u)
	}
	return
}

// V returns the right singular values
func (s svd) V() (V Mesher) {
	V = Gen(nil, s.n, s.n)
	copy(V.Slice(), s.v)
	return
}

// SingularVals returns the slice of singular values
func (s svd) SingularVals() (sv []float64) {
	return s.s
}

// S returns the diagonal matrix of singular values
func (s svd) S() (S Mesher) {
	if s.m < s.n {
		S = Gen(nil, s.m, s.n)
	} else {
		S = Gen(nil, s.n, s.n)
	}
	for i := 0; i < int(math.Min(float64(s.m), float64(s.n))); i++ {
		S.Set(s.s[i], i, i)
	}
	return
}

// Norm2 returns two norm - max(s)
func (s svd) Norm2() float64 {
	return s.s[0]
}

// Cond returns two norm condition number
// Max(s)/Min(s)
func (s svd) Cond() float64 {
	if s.m < s.n {
		return s.s[0] / s.s[len(s.s)-2]
	}
	return s.s[0] / s.s[len(s.s)-1]
}

// Rank computes the effective numerical matrix rank
// Number of nonnegligible singular values
func (s svd) Rank() int {
	tol := math.Max(float64(s.m), float64(s.n)) * s.s[0] * Eps
	r := 0
	for _, v := range s.s {
		if v > tol {
			r++
		}
	}
	return r
}
