use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Axis, Array};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::buffer::PyBuffer;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn rust_stats<'py>(py: Python<'py>, m: &'py PyModule) -> PyResult<()> {

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

    fn ks_2samp_buffer(data1buffer: PyBuffer<f64>, 
                       data2buffer: PyBuffer<f64>) -> f64 {

        if (data1buffer.dimensions() != 1) ||(data2buffer.dimensions() != 1) {
            panic!("data buffers not 1d!");
        }

        let mut mind = 0f64;
        let mut maxd = 0f64;
        
        Python::with_gil(|py| {
            let n1 = data1buffer.item_count();
            let n2 = data2buffer.item_count();
            let inv_n1 = 1f64 / (n1 as f64);
            let inv_n2 = 1f64 / (n2 as f64);

            let mut d = 0f64;

            let mut i = 0;
            let mut j = 0;

            let data1 = data1buffer.as_slice(py).unwrap();
            let data2 = data2buffer.as_slice(py).unwrap();

            while (i < n1) && (j < n2) {
                let d1i = data1[i].get();
                let d2j = data2[j].get();

                if d1i <= d2j {
                    while (i < n1) && (data1[i].get() == d1i) {
                        d += inv_n1;
                        i += 1;
                    }
                }

                if d1i >= d2j {
                    while (j < n2) && (data2[j].get() == d2j) {
                        d -= inv_n2;
                        j += 1;
                    }
                }
                if d < mind { mind = d; }
                if d > maxd { maxd = d; }
                //mind = mind.min(d);
                //maxd = maxd.max(d);
            }
        });

        maxd - mind
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

    #[pyfn(m)]
    #[pyo3(name = "ks_2samp_buffer")]
    fn ks_2samp_buffer_py<'py>(
        b1: PyBuffer<f64>,
        b2: PyBuffer<f64>,
    ) -> f64 {
        ks_2samp_buffer(b1, b2)
    }

    Ok(())
}