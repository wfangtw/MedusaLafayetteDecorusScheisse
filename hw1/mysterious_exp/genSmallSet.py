import numpy as np
INPUT_DIM = 39
OUTPUT_DIM = 48
DATA_PER_PHONE = 100

proto_x = np.random.randn(OUTPUT_DIM,INPUT_DIM) * 10
proto_y = np.zeros((OUTPUT_DIM,OUTPUT_DIM))
np.fill_diagonal(proto_y,1)
f = open('smallset.train', 'w')

output_x = (proto_x + np.random.randn(OUTPUT_DIM,INPUT_DIM)/20).tolist()
output_y = proto_y.tolist()
for i in range(1, DATA_PER_PHONE):
    instance_x = proto_x + np.random.randn(OUTPUT_DIM,INPUT_DIM)/20
    output_x.extend(instance_x.tolist())
    output_y.extend(proto_y.tolist())
print(len(output_x))
output = (output_x,output_y)
f.write(str(output))
f.close()

f = open('smallset.dev', 'w')
output_x = (proto_x + np.random.randn(OUTPUT_DIM,INPUT_DIM)/20).tolist()
output_y = proto_y.tolist()
for i in range(1, DATA_PER_PHONE/10):
    instance_x = proto_x + np.random.randn(OUTPUT_DIM,INPUT_DIM)/20
    output_x.extend(instance_x.tolist())
    output_y.extend(proto_y.tolist())
print(len(output_x))
output = (output_x,output_y)
f.write(str(output))
