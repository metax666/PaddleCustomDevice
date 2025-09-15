cmake_minimum_required(VERSION 3.10)
project(PythonUnitTests)

enable_testing()

find_package(Python REQUIRED COMPONENTS Interpreter)

set(PADDLE_LEGACY_TEST_PATH
    ${CMAKE_CURRENT_LIST_DIR}/../../../Paddle/test/legacy_test)
set(METAX_UNIT_TEST_PATH ${CMAKE_CURRENT_LIST_DIR}/unit_test)

file(GLOB_RECURSE PYTHON_TEST_SCRIPTS "${METAX_UNIT_TEST_PATH}/*.py")

list(
  APPEND
  PYTHON_TEST_SCRIPTS
  ${PADDLE_LEGACY_TEST_PATH}/test_accuracy_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_tril_triu_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_where_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_split_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_fill_constant_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_empty_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_sign_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_cast_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_index_add_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_unbind_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_put_along_axis_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_layer_norm_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_maximum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_accuracy_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_strided_slice_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_sum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_set_value_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_flatten_contiguous_range_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_top_k_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_subtract_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_softmax_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_cumsum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_greater_equal_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_div_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_top_k_v2_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_stack_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_one_hot_v2_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_fill_any_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_gather_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_reshape_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_index_put_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_bitwise_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_max_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_pad_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_pow_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_uniform_random_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_scatter_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_cast_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_zeros_like_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_compare_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_shape_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_tril_triu_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_slice_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_add_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_index_put_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_bincount_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_assign_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_logical_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_squared_l2_norm_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_mean_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_fused_bias_act_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_expand_v2_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_adamw_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_gather_nd_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_concat_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_scatter_nd_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_floordiv_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_mul_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_transpose_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_einsum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_randint_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_c_embedding_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_numel_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_scale_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_softmax_with_cross_entropy_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_full_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_scatter_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_clip_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_reduce_op.py)

list(
  REMOVE_ITEM
  PYTHON_TEST_SCRIPTS
  ${PADDLE_LEGACY_TEST_PATH}/test_sum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_cumsum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_softmax_with_cross_entropy_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_expand_v2_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_tril_triu_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_squared_l2_norm_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_softmax_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_index_add_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_div_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_stack_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_gather_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_scatter_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_logical_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_mean_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_transpose_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_einsum_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_c_embedding_op.py
  ${PADDLE_LEGACY_TEST_PATH}/test_layer_norm_op.py)

list(REMOVE_DUPLICATES PYTHON_TEST_SCRIPTS)
foreach(test_script ${PYTHON_TEST_SCRIPTS})
  get_filename_component(test_name ${test_script} NAME_WE)

  add_test(
    NAME "python_${test_name}"
    COMMAND ${Python_EXECUTABLE} ${test_script}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties("python_${test_name}" PROPERTIES TIMEOUT 360)
endforeach()
