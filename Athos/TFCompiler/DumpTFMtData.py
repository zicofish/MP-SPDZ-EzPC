'''

Authors: Nishant Kumar.

Copyright:
Copyright (c) 2018 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import numpy
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def save_graph_metadata(output_tensor, sess, feed_dict):
  #First save the graph def
  graph_def = tf.get_default_graph().as_graph_def()
  optimized_graph_def = graph_def
  with open('./graphDef.mtdata', 'w') as f:
    f.write(str(optimized_graph_def))
  with open('./graphDef.bin', 'wb') as f:
    f.write(optimized_graph_def.SerializeToString())

  # Save size information for tensors on which output depends
  tensors_to_evaluate = []
  tensors_to_evaluate_names = []
  graph = tf.get_default_graph()
  for node in optimized_graph_def.node:
    if graph.get_operation_by_name(node.name).outputs:
      cur_output = graph.get_operation_by_name(node.name).outputs[0]
      tensors_to_evaluate.append(cur_output)
      tensors_to_evaluate_names.append(node.name)
  tensors_evaluated = sess.run(tensors_to_evaluate, feed_dict)
  tensors_shape = list(map(lambda x : x.shape, tensors_evaluated))

  # Write size info in a file
  with open('./sizeInfo.mtdata','w') as f:
    for ii, curr in enumerate(tensors_to_evaluate_names):
      curShape = tensors_shape[ii]
      f.write(tensors_to_evaluate_names[ii] + ' ')
      for dim in curShape:
        f.write(str(dim)+' ')
      f.write('\n')

  return optimized_graph_def

def dumpImageDataInt(imgData, filename, scalingFac, writeMode):
  print("Dumping image data...")
  with open(filename, writeMode) as ff:
    numpy.array(imgData).tofile(ff)

def dumpInt(ff, tensor, scalingFac, sess, update=lambda x: x):
  tensor = sess.run(tensor)
  tensor.tofile(ff)

def dumpWeightsInt(filename, scalingFac, writeMode, sess):
  with open(filename, writeMode) as ff:
    for op in tf.get_default_graph().get_operations():
      if op.type in ('Conv2D', 'BiasAdd', 'MatMul'):
        dumpInt(ff, op.inputs[1], scalingFac, sess)
      elif op.type in ('FusedBatchNorm', 'FusedBatchNormV3'):
        gamma, beta, mu, variance = op.inputs[1:]

        epsilon = 1e-5 # Taken from non-fused BN of TF
        rsigma = tf.rsqrt(variance + epsilon)

        gamma = gamma * rsigma
        dumpInt(ff, gamma, scalingFac, sess)
        dumpInt(ff, beta - gamma * mu, scalingFac, sess)
        dumpInt(ff, tf.zeros(tf.shape(mu)), scalingFac, sess)
        dumpInt(ff, tf.fill(tf.shape(variance), 1-epsilon), scalingFac, sess)

def dumpImgAndWeightsData2(sess, imgData, filename, scalingFac):
  print("Starting to dump data...")
  dumpImageDataInt(imgData, filename, scalingFac, 'w')
  dumpWeightsInt(filename, scalingFac, 'a', sess)
