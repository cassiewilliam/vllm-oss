
## install the dependencies required for building mscclpp
git clone https://github.com/microsoft/mscclpp
mkdir mscclpp/build
cd mscclpp/build
cmake -DCMAKE_BUILD_TYPE=Release -DIBVERBS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libibverbs.so -DIBVERBS_INCLUDE_DIR=/usr/include ..
ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so
make -j
make install

cd ../python
python -m pip install mpi4py

pip install -r requirements_cuda12.txt
cd ..
# python -m pip install .