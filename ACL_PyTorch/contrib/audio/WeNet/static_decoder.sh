export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/bin:${install_path}/bin:${install_path}/atc/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/lib64:${install_path}/atc/lib64:${install_path}/acllib/lib64:${install_path}/compiler/lib64/plugin/opskernel:${install_path}/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/latest/python/site-packages:${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export ASCEND_AICPU_PATH=${install_path}
export ASCEND_OPP_PATH=${install_path}/opp
export TOOLCHAIN_HOME=${install_path}/toolkit
export ASCEND_AUTOML_PATH=${install_path}/tools
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}
atc --model=decoder_final.onnx --framework=5 --output=decoder_fendang --input_format=ND \
--input_shape="memory:10,-1,256;memory_mask:10,1,-1;ys_in_pad:10,-1;ys_in_lens:10;r_ys_in_pad:10,-1" --log=error  \
--dynamic_dims="96,96,3,3;96,96,4,4;96,96,5,5;96,96,6,6;96,96,7,7;96,96,8,8;96,96,9,9;96,96,10,10;96,96,11,11;\
96,96,12,12;96,96,13,13;96,96,14,14;96,96,15,15;96,96,16,16;96,96,17,17;96,96,18,18;96,96,19,19;96,96,20,20;\
96,96,21,21;96,96,22,22;96,96,23,23;144,144,6,6;144,144,7,7;144,144,8,8;144,144,9,9;144,144,10,10;144,144,11,11;\
144,144,12,12;144,144,13,13;144,144,14,14;144,144,15,15;144,144,16,16;144,144,17,17;144,144,18,18;144,144,19,19;\
144,144,20,20;144,144,21,21;144,144,22,22;144,144,23,23;144,144,24,24;144,144,25,25;144,144,26,26;144,144,27,27;\
144,144,28,28;384,384,9,9;384,384,10,10;384,384,11,11;384,384,12,12;384,384,13,13;384,384,14,14;384,384,15,15;\
384,384,16,16;384,384,17,17;384,384,18,18;384,384,19,19;384,384,20,20;384,384,21,21;384,384,22,22;384,384,23,23;\
384,384,24,24;384,384,25,25;384,384,26,26;384,384,27,27;384,384,28,28;384,384,29,29;384,384,30,30;384,384,31,31;\
384,384,32,32;384,384,33,33;384,384,34,34;384,384,35,35;384,384,36,36;384,384,37,37;384,384,38,38;384,384,39,39;384,384,40,40;384,384,41,41;" \
--soc_version=Ascend310

