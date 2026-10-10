import pytest
torch = pytest.importorskip("torch")
from vmd_eog.neural import RegionalBandRouter,DeploymentStudent,fourier_components,input_scale
from vmd_eog.neural_losses import paired_objective


def example(channels=5,length=512):
    torch.manual_seed(42)
    eeg = torch.randn(1,channels,length)
    mask = torch.ones(1,channels,dtype=torch.bool)
    regions = torch.tensor([[0,0,1,2,3][:channels]])
    sides = torch.tensor([[0,1,2,3,3][:channels]])
    return eeg,mask,{"regions":regions,"hemispheres":sides}


def make_nonidentity(model):
    with torch.no_grad():
        for head in model.heads.heads:
            head.weight.normal_(0,.01)


def test_fourier_partition_closes_without_frequency_gaps():
    eeg,_,_ = example()
    pieces,descriptors = fourier_components(eeg)
    torch.testing.assert_close(pieces.sum(-2),eeg,rtol=1e-5,atol=1e-6)
    assert descriptors.shape == (1,5,6,5)
    assert torch.isfinite(descriptors).all()


@pytest.mark.parametrize("factory",[lambda:RegionalBandRouter(variant="fixed"),DeploymentStudent])
def test_identity_and_nontrivial_channel_permutation(factory):
    eeg,mask,metadata = example()
    model = factory().eval()
    torch.testing.assert_close(model(eeg,mask,metadata)["cleaned"],eeg,rtol=0,atol=0)
    make_nonidentity(model)
    original = model(eeg,mask,metadata)["cleaned"]
    order = torch.tensor([4,2,0,3,1])
    shuffled = model(eeg[:,order],mask[:,order],{key:value[:,order] for key,value in metadata.items()})["cleaned"]
    torch.testing.assert_close(shuffled,original[:,order],rtol=2e-5,atol=2e-6)


def test_invalid_padding_and_missing_frontal_context_do_not_change_valid_output():
    eeg,mask,metadata = example()
    metadata["regions"].fill_(3)
    model = DeploymentStudent().eval()
    make_nonidentity(model)
    original = model(eeg,mask,metadata)["cleaned"]
    padded_eeg = torch.cat((eeg,torch.full((1,2,eeg.shape[-1]),float('nan'))),1)
    padded_mask = torch.cat((mask,torch.zeros(1,2,dtype=torch.bool)),1)
    padded_metadata = {key:torch.cat((value,torch.full((1,2),3)),1) for key,value in metadata.items()}
    result = model(padded_eeg,padded_mask,padded_metadata)["cleaned"]
    torch.testing.assert_close(result[:,:5],original)


def test_component_permutation_with_descriptors_and_masks():
    eeg,mask,metadata = example()
    components,descriptors = fourier_components(eeg)
    bundle = {"components":components,"descriptors":descriptors,"valid":torch.ones(1,5,6,dtype=torch.bool),"scale":input_scale(eeg).unsqueeze(-2)}
    model = RegionalBandRouter(variant="all_vmd",component_dropout=0).eval()
    make_nonidentity(model)
    first = model(eeg,mask,metadata,bundle)["cleaned"]
    order = torch.tensor([4,2,5,0,3,1])
    second_bundle = {"components":components[:,:,order],"descriptors":descriptors[:,:,order],"valid":bundle["valid"][:,:,order],"scale":bundle["scale"]}
    second = model(eeg,mask,metadata,second_bundle)["cleaned"]
    torch.testing.assert_close(first,second,rtol=2e-5,atol=3e-6)


def test_paired_error_does_not_reward_zeroing_eeg():
    eeg,mask,_ = example()
    config = {"fs":200,"neural":{"identity_weight":1.,"psd_weight":.1,"covariance_weight":.05,"snr_weight":.01}}
    clean = torch.ones(1,dtype=torch.bool)
    exact,_ = paired_objective(eeg,eeg,eeg,mask,clean,config,"snr")
    erased,_ = paired_objective(torch.zeros_like(eeg),eeg,eeg,mask,clean,config,"snr")
    assert exact < erased and torch.isfinite(erased)
