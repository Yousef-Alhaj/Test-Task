from main import torch,np
from Split import test_mean,test_std
from train import  model,device
from dataSet import X_test

model.eval()
with torch.no_grad():
    y_pred = []
    for i in range(len(X_test)):
        input_seq = torch.Tensor(X_test[i]).unsqueeze(0).unsqueeze(1).to(device)
        output = model(input_seq)
        y_pred.append(output.item())

y_pred = (np.array(y_pred) * test_std) + test_mean

print("Predicted Ozone Value:")
print(y_pred[-1])
