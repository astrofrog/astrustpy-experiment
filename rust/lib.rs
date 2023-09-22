use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Axis};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::PyObject;
use pyo3::{pymodule, types::PyModule, PyResult, Python};


#[pymodule]
fn rust_stats<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>){
        x *= a;
        println!("first is {}", x[0]);
        x[0] = x[0] + 4f64;
    }

    fn ks_2samp(data1: ArrayViewD<'_, f64>, 
                data2: ArrayViewD<'_, f64>) -> f64 {

        let n1 = data1.len_of(Axis(0));
        let n2 = data2.len_of(Axis(0));
        let inv_n1 = 1f64 / (n1 as f64);
        let inv_n2 = 1f64 / (n2 as f64);
    
        let mut d = 0f64;
        let mut mind = 0f64;
        let mut maxd = 0f64;

        let mut i = 0;
        let mut j = 0;

        while (i < n1) && (j < n2) {
            let d1i = data1[i];
            let d2j = data2[j];

            if d1i <= d2j {
                while (i < n1) && (data1[i] == d1i) {
                    d += inv_n1;
                    i += 1;
                }
            }

            if d1i >= d2j {
                while (j < n2) && (data2[j] == d2j) {
                    d -= inv_n2;
                    j += 1;
                }
            }

            mind = mind.min(d);
            maxd = maxd.max(d);
        }

        maxd - mind
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py<'py>(a: f64, x: &'py PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }


    #[pyfn(m)]
    fn print_rust() -> PyResult<()> {
        println!("test");
        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "ks_2samp")]
    fn ks_2samp_py<'py>(
        data1: PyReadonlyArrayDyn<'py, f64>,
        data2: PyReadonlyArrayDyn<'py, f64>,
    ) -> f64 {
        let data1 = data1.as_array();
        let data2 = data2.as_array();
        ks_2samp(data1, data2)
    }

    Ok(())
}