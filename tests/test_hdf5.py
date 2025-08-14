import h5py
import pytest 
import numpy
 
@pytest.mark.parametrize("dataset_id", ["Suv_RA","gammaAB_RA","gradient_nuclear","gradient_electronic","gradient"])
@pytest.mark.parametrize("file_id", ["H2","H2O","HF","HO"])
@pytest.mark.parametrize("question_id", ["hw5_2"])
def test_dataset(question_id, file_id, dataset_id, sample_output_dir, student_output_dir):
    """Test whether dataset values in reference and student outputs match."""

    reference_file = sample_output_dir / question_id / (file_id + ".hdf5")
    test_file = student_output_dir / question_id / (file_id + ".hdf5")
    
    assert reference_file.exists() and reference_file.is_file()
    assert test_file.exists and test_file.is_file() 

    with h5py.File(reference_file, "r") as ref, h5py.File(test_file, "r") as test:
        ref_data = ref[dataset_id][()]
        test_data = test[dataset_id][()]

        assert ref_data.shape == test_data.shape, (
            f"Shape of reference data: {ref_data.shape} does not match shape of test data {test_data.shape}"
        )
        assert numpy.allclose(ref_data, test_data, rtol=1e-4, atol=1e-2), (
            f"Dataset {dataset_id} differs in {reference_file} vs {test_file}"
        )
