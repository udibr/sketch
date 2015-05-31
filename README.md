# sketch
Use RNN to model handwriting using Theano blocks.
I am trying to reproduce handwriting model given by
[Alex Graves](http://arxiv.org/abs/1308.0850),
see also his [demo](http://www.cs.toronto.edu/~graves/handwriting.html)

Dependencies
------------
* [Blocks](https://github.com/bartvm/blocks) follow
  the [install instructions](http://blocks.readthedocs.org/en/latest/setup.html).
  This will install all the other dependencies for you (Theano, Fuel, etc.).
* Download handwriting dataset using [this notebook](./handwriting-to-hdf5.ipynb)
 
Running code
------------

    python sketch.py
The run parameteres are used to build a run name which is used to 
create a directory where the model is saved and also samples of handwriting
`samples-sketch.png`

For example:

```bash
python sketch.py --dim 900 --depth 1 --bs 56 --lr 1e-3  -G 10
         Epoch 1, step 105
         test_sequence_log_likelihood: -631.519287109
python sketch.py --dim 900 --depth 1 --bs 56 --lr 3e-4  -G 10 --model handwriting-1X900m20d5r13b56e15G10
         Epoch 5, step 25
         test_sequence_log_likelihood: -950.866210938
python sketch.py --dim 900 --depth 1 --bs 56 --lr 1e-4  -G 10 --model handwriting-1X900m20d5r34b56e15G10
         Epoch 144, step 120
         test_sequence_log_likelihood: -1481.65966797
```

 ![samples](sketch.png)


Notes
-----
* This is a work in progress


python sketch.py --dim 900 --depth 1 --bs 56 --lr 1e-4  -G 10 --model handwriting-1X900m20d5r14b56e15G10 --sample