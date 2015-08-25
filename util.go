package mesh

func VecMax(v []float64) (i int, m float64) {
	m = v[0]
	for r, e := range v {
		if e > m {
			i = r
			m = e
		}
	}
	return
}

func VecMin(v []float64) (i int, m float64) {
	m = v[0]
	for r, e := range v {
		if e < m {
			i = r
			m = e
		}
	}
	return
}

// Clear zeros the vector, v
func Clear(v []float64) {
	for j := range v {
		v[j] = 0
	}
}
