"""Kiểm tra số học, gradient và tính nhân quả, không huấn luyện lại."""
import ast
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import torch
import models
from losses import digitclassifier_loss,languageid_loss,digitconvolution_Loss

torch.set_num_threads(2)
torch.manual_seed(7)
checks={}
for shape,kshape in [((3,3),(2,2)),((4,6),(2,3)),((6,5),(3,2))]:
    x=torch.randn(*shape,dtype=torch.float64,requires_grad=True)
    k=torch.randn(*kshape,dtype=torch.float64,requires_grad=True)
    actual=models.Convolve(x,k)
    expected=torch.nn.functional.conv2d(x[None,None],k[None,None])[0,0]
    assert torch.allclose(actual,expected,atol=1e-10)
    actual.square().sum().backward()
    assert x.grad is not None and k.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(k.grad).all()
checks['convolution_rectangles_values_and_gradients']=True

model=models.LanguageIDModel()
x=torch.nn.functional.one_hot(torch.randint(0,47,(3,2)),47).float()
padded=torch.cat([x,torch.zeros(4,2,47)])
assert torch.allclose(model(x),model(padded),atol=1e-6)
checks['rnn_zero_padding_keeps_output']=True

for cls,dim,lossfn in [(models.DigitClassificationModel,784,digitclassifier_loss),
                       (models.DigitConvolutionalModel,784,digitconvolution_Loss)]:
    model=cls(); x=torch.rand(2,dim,requires_grad=True)
    y=torch.nn.functional.one_hot(torch.tensor([0,3]),10).float().requires_grad_()
    scores=model(x); assert scores.shape==(2,10)
    loss=lossfn(scores,y); loss.backward()
    assert x.grad is not None and y.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
checks['q3_q5_shapes_and_gradients']=True

for B,T,C,block in [(1,1,2,4),(2,3,8,6),(3,6,4,6)]:
    model=models.Attention(C,block)
    x=torch.randn(B,T,C,requires_grad=True); y=model(x)
    reference=torch.nn.functional.scaled_dot_product_attention(
        model.q_layer(x),model.k_layer(x),model.v_layer(x),is_causal=True)
    assert y.shape==(B,T,C) and torch.allclose(y,reference,atol=1e-6,rtol=1e-5)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    if T>1:
        changed=x.detach().clone(); changed[:,-1]+=100
        assert torch.allclose(y[:,:-1],model(changed)[:,:-1],atol=1e-6)
checks['attention_reference_gradients_and_causality']=True

for name in ['models.py','train.py','losses.py']:
    tree=ast.parse((ROOT/name).read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Pass) for n in ast.walk(tree))
checks['no_unimplemented_pass']=True
(ROOT/'ket_qua'/'kiem_thu_bo_sung.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
