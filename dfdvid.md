install conda

2. Prepare Environment

initialize conda in terminal

conda init --all

create the environment

conda create --name dfdvid python=3.10

install accelerator

// Find some for 3.10
conda install conda-forge::cuda-runtime=12.4.1 conda-forge::cudnn=9.2.1.18

3. Install the requirements

pip
(dfdVid) C:\Users\admin\Documents\df\Detector\df-detector-vid>pip list
Package                      Version
---------------------------- --------------
absl-py                      2.1.0
anyio                        4.6.2.post1
argon2-cffi                  23.1.0
argon2-cffi-bindings         21.2.0
arrow                        1.3.0
asttokens                    2.4.1
astunparse                   1.6.3
async-lru                    2.0.4
attrs                        24.2.0
babel                        2.16.0
beautifulsoup4               4.12.3
bleach                       6.2.0
Brotli                       1.0.9
cachetools                   5.5.0
certifi                      2024.8.30
cffi                         1.17.1
charset-normalizer           3.3.2
colorama                     0.4.6
comm                         0.2.2
contourpy                    1.3.0
cycler                       0.12.1
debugpy                      1.8.7
decorator                    4.4.2
decord                       0.6.0
defusedxml                   0.7.1
dlib                         19.24.6
exceptiongroup               1.2.2
executing                    2.1.0
facenet-pytorch              2.5.3
fastjsonschema               2.20.0
filelock                     3.13.1
flatbuffers                  24.3.25
fonttools                    4.54.1
fqdn                         1.5.1
fsspec                       2024.10.0
gast                         0.4.0
gmpy2                        2.1.2
google-auth                  2.36.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.67.1
h11                          0.14.0
h5py                         3.12.1
httpcore                     1.0.6
httpx                        0.27.2
idna                         3.7
imageio                      2.36.0
imageio-ffmpeg               0.5.1
importlib_metadata           8.5.0
importlib_resources          6.4.5
ipykernel                    6.29.5
ipython                      8.18.1
ipywidgets                   8.1.5
isoduration                  20.11.0
jedi                         0.19.1
Jinja2                       3.1.4
joblib                       1.4.2
json5                        0.9.25
jsonpointer                  3.0.0
jsonschema                   4.23.0
jsonschema-specifications    2024.10.1
jupyter_client               8.6.3
jupyter_core                 5.7.2
jupyter-events               0.10.0
jupyter-lsp                  2.2.5
jupyter_server               2.14.2
jupyter_server_terminals     0.5.3
jupyterlab                   4.3.0
jupyterlab_pygments          0.3.0
jupyterlab_server            2.27.3
jupyterlab_widgets           3.0.13
keras                        2.10.0
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.7
libclang                     18.1.1
llvmlite                     0.43.0
Markdown                     3.7
markdown-it-py               3.0.0
MarkupSafe                   2.1.3
matplotlib                   3.9.2
matplotlib-inline            0.1.7
mdurl                        0.1.2
mistune                      3.0.2
mkl_fft                      1.3.11
mkl_random                   1.2.8
mkl-service                  2.4.0
ml-dtypes                    0.4.1
moviepy                      1.0.3
mpmath                       1.3.0
namex                        0.0.8
nbclient                     0.10.0
nbconvert                    7.16.4
nbformat                     5.10.4
nest-asyncio                 1.6.0
networkx                     3.2.1
notebook_shim                0.2.4
numba                        0.60.0
numpy                        1.26.4
oauthlib                     3.2.2
opencv-python                4.10.0.84
opt_einsum                   3.4.0
optree                       0.13.0
overrides                    7.7.0
packaging                    24.1
pandas                       2.2.3
pandocfilters                1.5.1
parso                        0.8.4
pillow                       10.4.0
pip                          24.2
platformdirs                 4.3.6
proglog                      0.1.10
prometheus_client            0.21.0
prompt_toolkit               3.0.48
protobuf                     3.19.6
psutil                       6.1.0
pure_eval                    0.2.3
pyasn1                       0.6.1
pyasn1_modules               0.4.1
pycparser                    2.22
Pygments                     2.18.0
pyparsing                    3.2.0
PySocks                      1.7.1
python-dateutil              2.9.0.post0
python-json-logger           2.0.7
pytz                         2024.2
pywin32                      308
pywinpty                     2.0.14
PyYAML                       6.0.2
pyzmq                        26.2.0
referencing                  0.35.1
requests                     2.32.3
requests-oauthlib            2.0.0
rfc3339-validator            0.1.4
rfc3986-validator            0.1.1
rich                         13.9.4
rpds-py                      0.21.0
rsa                          4.9
scikit-learn                 1.5.2
scipy                        1.13.1
seaborn                      0.13.2
Send2Trash                   1.8.3
setuptools                   75.1.0
six                          1.16.0
sniffio                      1.3.1
soupsieve                    2.6
stack-data                   0.6.3
sympy                        1.13.1
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.10.1
tensorflow-estimator         2.10.0
tensorflow_intel             2.18.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    2.5.0
terminado                    0.18.1
threadpoolctl                3.5.0
tinycss2                     1.4.0
tomli                        2.0.2
torch                        2.5.1
torchaudio                   2.5.1
torchvision                  0.20.1
tornado                      6.4.1
tqdm                         4.67.0
traitlets                    5.14.3
types-python-dateutil        2.9.0.20241003
typing_extensions            4.11.0
tzdata                       2024.2
uri-template                 1.3.0
urllib3                      2.2.3
wcwidth                      0.2.13
webcolors                    24.8.0
webencodings                 0.5.1
websocket-client             1.8.0
Werkzeug                     3.1.2
wheel                        0.44.0
widgetsnbextension           4.0.13
win-inet-pton                1.1.0
wrapt                        1.16.0
zipp                         3.20.2

conda
# packages in environment at C:\Users\admin\MiniConda3\envs\dfdVid:
#
# Name                    Version                   Build  Channel
absl-py                   2.1.0                    pypi_0    pypi
anyio                     4.6.2.post1              pypi_0    pypi
argon2-cffi               23.1.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.3.0                    pypi_0    pypi
asttokens                 2.4.1                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
async-lru                 2.0.4                    pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
babel                     2.16.0                   pypi_0    pypi
beautifulsoup4            4.12.3                   pypi_0    pypi
blas                      1.0                         mkl  
bleach                    6.2.0                    pypi_0    pypi
brotli-python             1.0.9            py39hd77b12b_8  
ca-certificates           2024.9.24            haa95532_0
cachetools                5.5.0                    pypi_0    pypi
certifi                   2024.8.30        py39haa95532_0
cffi                      1.17.1                   pypi_0    pypi
charset-normalizer        3.3.2              pyhd3eb1b0_0
colorama                  0.4.6                    pypi_0    pypi
comm                      0.2.2                    pypi_0    pypi
contourpy                 1.3.0                    pypi_0    pypi
cuda-cccl                 12.6.77                       0    nvidia
cuda-cccl_win-64          12.6.77                       0    nvidia
cuda-cudart               11.8.89                       0    nvidia
cuda-cudart-dev           11.8.89                       0    nvidia
cuda-cupti                11.8.87                       0    nvidia
cuda-libraries            11.8.0                        0    nvidia
cuda-libraries-dev        11.8.0                        0    nvidia
cuda-nvrtc                11.8.89                       0    nvidia
cuda-nvrtc-dev            11.8.89                       0    nvidia
cuda-nvtx                 11.8.86                       0    nvidia
cuda-profiler-api         12.6.77                       0    nvidia
cuda-runtime              11.8.0                        0    nvidia
cuda-version              12.6                          3    nvidia
cycler                    0.12.1                   pypi_0    pypi
debugpy                   1.8.7                    pypi_0    pypi
decorator                 4.4.2                    pypi_0    pypi
decord                    0.6.0                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
dlib                      19.24.6                  pypi_0    pypi
exceptiongroup            1.2.2                    pypi_0    pypi
executing                 2.1.0                    pypi_0    pypi
facenet-pytorch           2.5.3                    pypi_0    pypi
fastjsonschema            2.20.0                   pypi_0    pypi
filelock                  3.13.1           py39haa95532_0
flatbuffers               24.3.25                  pypi_0    pypi
fonttools                 4.54.1                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.12.1               ha860e81_0
fsspec                    2024.10.0                pypi_0    pypi
gast                      0.4.0                    pypi_0    pypi
giflib                    5.2.2                h7edc060_0
gmpy2                     2.1.2            py39h7f96b67_0
google-auth               2.36.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.67.1                   pypi_0    pypi
h11                       0.14.0                   pypi_0    pypi
h5py                      3.12.1                   pypi_0    pypi
httpcore                  1.0.6                    pypi_0    pypi
httpx                     0.27.2                   pypi_0    pypi
idna                      3.7              py39haa95532_0
imageio                   2.36.0                   pypi_0    pypi
imageio-ffmpeg            0.5.1                    pypi_0    pypi
importlib-metadata        8.5.0                    pypi_0    pypi
importlib-resources       6.4.5                    pypi_0    pypi
intel-openmp              2023.1.0         h59b6b97_46320
ipykernel                 6.29.5                   pypi_0    pypi
ipython                   8.18.1                   pypi_0    pypi
ipywidgets                8.1.5                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.19.1                   pypi_0    pypi
jinja2                    3.1.4            py39haa95532_1
joblib                    1.4.2                    pypi_0    pypi
jpeg                      9e                   h827c3e9_3
json5                     0.9.25                   pypi_0    pypi
jsonpointer               3.0.0                    pypi_0    pypi
jsonschema                4.23.0                   pypi_0    pypi
jsonschema-specifications 2024.10.1                pypi_0    pypi
jupyter-client            8.6.3                    pypi_0    pypi
jupyter-core              5.7.2                    pypi_0    pypi
jupyter-events            0.10.0                   pypi_0    pypi
jupyter-lsp               2.2.5                    pypi_0    pypi
jupyter-server            2.14.2                   pypi_0    pypi
jupyter-server-terminals  0.5.3                    pypi_0    pypi
jupyterlab                4.3.0                    pypi_0    pypi
jupyterlab-pygments       0.3.0                    pypi_0    pypi
jupyterlab-server         2.27.3                   pypi_0    pypi
jupyterlab-widgets        3.0.13                   pypi_0    pypi
keras                     2.10.0                   pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
kiwisolver                1.4.7                    pypi_0    pypi
lcms2                     2.12                 h83e58a3_0
lerc                      3.0                  hd77b12b_0
libclang                  18.1.1                   pypi_0    pypi
libcublas                 11.11.3.6                     0    nvidia
libcublas-dev             11.11.3.6                     0    nvidia
libcufft                  10.9.0.58                     0    nvidia
libcufft-dev              10.9.0.58                     0    nvidia
libcurand                 10.3.7.77                     0    nvidia
libcurand-dev             10.3.7.77                     0    nvidia
libcusolver               11.4.1.48                     0    nvidia
libcusolver-dev           11.4.1.48                     0    nvidia
libcusparse               11.7.5.86                     0    nvidia
libcusparse-dev           11.7.5.86                     0    nvidia
libdeflate                1.17                 h2bbff1b_1
libjpeg-turbo             2.0.0                h196d8e1_0
libnpp                    11.8.0.86                     0    nvidia
libnpp-dev                11.8.0.86                     0    nvidia
libnvjpeg                 11.9.0.86                     0    nvidia
libnvjpeg-dev             11.9.0.86                     0    nvidia
libpng                    1.6.39               h8cc25b3_0
libtiff                   4.5.1                hd77b12b_0
libuv                     1.48.0               h827c3e9_0
libwebp                   1.3.2                hbc33d0d_0
libwebp-base              1.3.2                h3d04722_1
llvmlite                  0.43.0                   pypi_0    pypi
lz4-c                     1.9.4                h2bbff1b_1
markdown                  3.7                      pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.3            py39h2bbff1b_0
matplotlib                3.9.2                    pypi_0    pypi
matplotlib-inline         0.1.7                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mistune                   3.0.2                    pypi_0    pypi
mkl                       2023.1.0         h6b88ed4_46358
mkl-service               2.4.0            py39h2bbff1b_1
mkl_fft                   1.3.11           py39h827c3e9_0
mkl_random                1.2.8            py39hc64d2fc_0
ml-dtypes                 0.4.1                    pypi_0    pypi
moviepy                   1.0.3                    pypi_0    pypi
mpc                       1.1.0                h7edee0f_1
mpfr                      4.0.2                h62dcd97_1
mpir                      3.0.0                hec2e145_1
mpmath                    1.3.0            py39haa95532_0
namex                     0.0.8                    pypi_0    pypi
nbclient                  0.10.0                   pypi_0    pypi
nbconvert                 7.16.4                   pypi_0    pypi
nbformat                  5.10.4                   pypi_0    pypi
nest-asyncio              1.6.0                    pypi_0    pypi
networkx                  3.2.1            py39haa95532_0
notebook-shim             0.2.4                    pypi_0    pypi
numba                     0.60.0                   pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
opencv-python             4.10.0.84                pypi_0    pypi
openjpeg                  2.5.2                hae555c5_0
openssl                   3.0.15               h827c3e9_0
opt-einsum                3.4.0                    pypi_0    pypi
optree                    0.13.0                   pypi_0    pypi
overrides                 7.7.0                    pypi_0    pypi
packaging                 24.1                     pypi_0    pypi
pandas                    2.2.3                    pypi_0    pypi
pandocfilters             1.5.1                    pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
pillow                    10.4.0           py39h827c3e9_0
pip                       24.2             py39haa95532_0
platformdirs              4.3.6                    pypi_0    pypi
proglog                   0.1.10                   pypi_0    pypi
prometheus-client         0.21.0                   pypi_0    pypi
prompt-toolkit            3.0.48                   pypi_0    pypi
protobuf                  3.19.6                   pypi_0    pypi
psutil                    6.1.0                    pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
pyasn1                    0.6.1                    pypi_0    pypi
pyasn1-modules            0.4.1                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pygments                  2.18.0                   pypi_0    pypi
pyparsing                 3.2.0                    pypi_0    pypi
pysocks                   1.7.1            py39haa95532_0
python                    3.9.20               h8205438_1
python-dateutil           2.9.0.post0              pypi_0    pypi
python-json-logger        2.0.7                    pypi_0    pypi
pytorch                   2.5.1           py3.9_cuda11.8_cudnn9_0    pytorch
pytorch-cuda              11.8                 h24eeafa_6    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2024.2                   pypi_0    pypi
pywin32                   308                      pypi_0    pypi
pywinpty                  2.0.14                   pypi_0    pypi
pyyaml                    6.0.2            py39h827c3e9_0
pyzmq                     26.2.0                   pypi_0    pypi
referencing               0.35.1                   pypi_0    pypi
requests                  2.32.3           py39haa95532_0
requests-oauthlib         2.0.0                    pypi_0    pypi
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
rich                      13.9.4                   pypi_0    pypi
rpds-py                   0.21.0                   pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
scikit-learn              1.5.2                    pypi_0    pypi
scipy                     1.13.1                   pypi_0    pypi
seaborn                   0.13.2                   pypi_0    pypi
send2trash                1.8.3                    pypi_0    pypi
setuptools                75.1.0           py39haa95532_0
six                       1.16.0                   pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
soupsieve                 2.6                      pypi_0    pypi
sqlite                    3.45.3               h2bbff1b_0
stack-data                0.6.3                    pypi_0    pypi
sympy                     1.13.1                   pypi_0    pypi
tbb                       2021.8.0             h59b6b97_0
tensorboard               2.10.1                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow                2.10.1                   pypi_0    pypi
tensorflow-estimator      2.10.0                   pypi_0    pypi
tensorflow-intel          2.18.0                   pypi_0    pypi
tensorflow-io-gcs-filesystem 0.31.0                   pypi_0    pypi
termcolor                 2.5.0                    pypi_0    pypi
terminado                 0.18.1                   pypi_0    pypi
threadpoolctl             3.5.0                    pypi_0    pypi
tinycss2                  1.4.0                    pypi_0    pypi
tomli                     2.0.2                    pypi_0    pypi
torchaudio                2.5.1                    pypi_0    pypi
torchvision               0.20.1                   pypi_0    pypi
tornado                   6.4.1                    pypi_0    pypi
tqdm                      4.67.0                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
types-python-dateutil     2.9.0.20241003           pypi_0    pypi
typing_extensions         4.11.0           py39haa95532_0
tzdata                    2024.2                   pypi_0    pypi
uri-template              1.3.0                    pypi_0    pypi
urllib3                   2.2.3            py39haa95532_0
vc                        14.40                h2eaa2aa_1
vs2015_runtime            14.40.33807          h98bb1dd_1
wcwidth                   0.2.13                   pypi_0    pypi
webcolors                 24.8.0                   pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.8.0                    pypi_0    pypi
werkzeug                  3.1.2                    pypi_0    pypi
wheel                     0.44.0           py39haa95532_0
widgetsnbextension        4.0.13                   pypi_0    pypi
win_inet_pton             1.1.0            py39haa95532_0
wrapt                     1.16.0                   pypi_0    pypi
xz                        5.4.6                h8cc25b3_1
yaml                      0.2.5                he774522_0
zipp                      3.20.2                   pypi_0    pypi
zlib                      1.2.13               h8cc25b3_1
zstd                      1.5.6                h8880b57_0
