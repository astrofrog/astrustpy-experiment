use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn print_rust(text: &str) -> PyResult<()> {
    println!("{}", text);
    Ok(())
}

#[pymodule]
fn stats(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(print_rust))?;
    Ok(())
}
