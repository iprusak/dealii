// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2023 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#ifndef dealii_portable_fe_evaluation_h
#define dealii_portable_fe_evaluation_h

#include <deal.II/base/config.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_evaluation_kernels.h>
#include <deal.II/matrix_free/portable_hanging_nodes_internal.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>
#include <deal.II/matrix_free/portable_tensor_product_kernels.h>

#include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

/**
 * Namespace for portable capabilities
 */
namespace Portable
{
  /**
   * This class provides all the functions necessary to evaluate functions at
   * quadrature points and cell integrations. In functionality, this class is
   * similar to FEValues<dim>.
   *
   * This class has five template arguments:
   *
   * @tparam dim Dimension in which this class is to be used
   *
   * @tparam fe_degree Degree of the tensor product finite element with fe_degree+1
   * degrees of freedom per coordinate direction
   *
   * @tparam n_q_points_1d Number of points in the quadrature formular in 1d,
   * defaults to fe_degree+1
   *
   * @tparam n_components Number of vector components when solving a system of
   * PDEs. If the same operation is applied to several components of a PDE (e.g.
   * a vector Laplace equation), they can be applied simultaneously with one
   * call (and often more efficiently). Defaults to 1
   *
   * @tparam Number Number format, @p double or @p float. Defaults to @p
   * double.
   */
  template <int dim,
            int fe_degree,
            int n_q_points_1d = fe_degree + 1,
            int n_components_ = 1,
            typename Number   = double>
  class FEEvaluation
  {
  public:
    /**
     * An alias for the value type. This is @p Number for scalar problems
     * and Tensor<1, n_components> for vector-valued problems.
     */
    using value_type = std::conditional_t<(n_components_ == 1),
                                          Number,
                                          Tensor<1, n_components_, Number>>;

    /**
     * An alias for the gradient type.
     */
    using gradient_type = std::conditional_t<
      n_components_ == 1,
      Tensor<1, dim, Number>,
      std::conditional_t<n_components_ == dim,
                         Tensor<2, dim, Number>,
                         Tensor<1, n_components_, Tensor<1, dim, Number>>>>;

    /**
     * An alias to kernel specific information.
     */
    using data_type = typename MatrixFree<dim, Number>::Data;

    /**
     * Dimension.
     */
    static constexpr unsigned int dimension = dim;

    /**
     * Number of components.
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * Number of quadrature points per cell.
     */
    static constexpr unsigned int n_q_points =
      Utilities::pow(n_q_points_1d, dim);

    /**
     * Number of tensor degrees of freedom of a scalar component determined
     * from the given template argument `fe_degree`.
     */
    static constexpr unsigned int tensor_dofs_per_component =
      Utilities::pow(fe_degree + 1, dim);

    /**
     * Number of tensor degrees of freedom of all component determined from
     * the given template argument `fe_degree`. This is the total number of
     * local DoFs in a cell.
     */
    static constexpr unsigned int tensor_dofs_per_cell =
      tensor_dofs_per_component * n_components;

    /**
     * Constructor. You will need to provide a pointer to the
     * Portable::MatrixFree::Data object, which is typically provided to the
     * functor inside the
     * Portable::MatrixFree::cell_loop() and the index @p dof_handler_index
     * of the DoFHandler if more than one was provided when the
     * Portable::MatrixFree object was initialized.
     */
    DEAL_II_HOST_DEVICE
    explicit FEEvaluation(const data_type   *data,
                          const unsigned int dof_handler_index = 0);

    /**
     * Return the index of the current cell.
     */
    DEAL_II_HOST_DEVICE
    int
    get_current_cell_index();

    /**
     * Return a pointer to the MatrixFree<dim, Number>::Data object on device
     * that contains necessary constraint, dof index, and shape function
     * information for evaluation used in the matrix-free kernels.
     */
    DEAL_II_HOST_DEVICE
    const data_type *
    get_matrix_free_data();

    /**
     * For the vector @p src, read out the values on the degrees of freedom of
     * the current cell, and store them internally. Similar functionality as
     * the function DoFAccessor::get_interpolated_dof_values when no
     * constraints are present, but it also includes constraints from hanging
     * nodes, so one can see it as a similar function to
     * AffineConstraints::read_dof_values() as well.
     */
    DEAL_II_HOST_DEVICE void
    read_dof_values(const DeviceVector<Number> &src);

    /**
     * Take the value stored internally on dof values of the current cell and
     * sum them into the vector @p dst. The function also applies constraints
     * during the write operation. The functionality is hence similar to the
     * function AffineConstraints::distribute_local_to_global.
     */
    DEAL_II_HOST_DEVICE void
    distribute_local_to_global(DeviceVector<Number> &dst) const;

    /**
     * Evaluate the function values and the gradients of the FE function given
     * at the DoF values in the input vector at the quadrature points on the
     * unit cell. The function argument @p evaluate_flag specifies which parts
     * shall actually be computed. This function needs to be called before the
     * functions @p get_value() or @p get_gradient() give useful information.
     */
    DEAL_II_HOST_DEVICE void
    evaluate(const EvaluationFlags::EvaluationFlags evaluate_flag);

    /**
     * This function takes the values and/or gradients that are stored on
     * quadrature points, tests them by all the basis functions/gradients on
     * the cell and performs the cell integration as specified by the
     * @p integration_flag argument.
     */
    DEAL_II_HOST_DEVICE void
    integrate(const EvaluationFlags::EvaluationFlags integration_flag);

    /**
     * Return the value of the finite element function at the quadrature point
     * with index @p q_point after a call to evaluate() with
     * EvaluationFlags::values set.
     */
    DEAL_II_HOST_DEVICE value_type
    get_value(const int q_point) const;

    /**
     * Return the value stored for the local degree of freedom with index
     * @p dof_index. This accesses the data loaded by read_dof_values().
     */
    DEAL_II_HOST_DEVICE value_type
    get_dof_value(const int dof_index) const;

    /**
     * Submit the value @p val_in at quadrature point @p q_point for
     * subsequent integration via integrate() with EvaluationFlags::values
     * set.
     */
    DEAL_II_HOST_DEVICE void
    submit_value(const value_type &val_in, const int q_point);

    /**
     * Submit the value @p value for the local degree of freedom with index
     * @p dof_index, to be written out by a subsequent call to
     * distribute_local_to_global().
     */
    DEAL_II_HOST_DEVICE void
    submit_dof_value(const value_type &value, const int dof_index);

    /**
     * Return the gradient of the finite element function at the quadrature
     * point with index @p q_point after a call to evaluate() with
     * EvaluationFlags::gradients set.
     */
    DEAL_II_HOST_DEVICE gradient_type
    get_gradient(const int q_point) const;

    /**
     * Submit the gradient @p gradient at quadrature point @p q_point for
     * subsequent integration via integrate() with EvaluationFlags::gradients
     * set.
     */
    DEAL_II_HOST_DEVICE void
    submit_gradient(const gradient_type &gradient, const int q_point);

    /**
     * Return the symmetric gradient of the finite element function at
     * quadrature point @p q_point after a call to evaluate() with
     * EvaluationFlags::gradients set. This function is only available when
     * the number of components equals the dimension (n_components==dim).
     */
    DEAL_II_HOST_DEVICE
    SymmetricTensor<2, dim, Number>
    get_symmetric_gradient(const int q_point) const;

    /**
     * Return the divergence of the vector-valued finite element function at
     * quadrature point @p q_point after a call to evaluate() with
     * EvaluationFlags::gradients set. This function is only available when the
     * number of components equals the dimension (n_components==dim).
     */
    DEAL_II_HOST_DEVICE Number
    get_divergence(const int q_point) const;

    /**
     * Write a contribution that is multiplied by the divergence of the test
     * function to the field containing the gradients at quadrature point
     * @p q_point for subsequent integration via integrate() with
     * EvaluationFlags::gradients set. See submit_gradient() for further
     * information. This function is only available when the number of
     * components equals the dimension (n_components==dim).
     *
     * @note This operation writes the data to the same field as
     * submit_gradient() and submit_symmetric_gradient(). As a consequence,
     * only one of these functions can be used. In case several terms of this
     * kind appear in a weak form, the contribution of a potential call to this
     * function must be added into the diagonal of the rank-2 tensor
     * contribution passed to submit_gradient().
     */
    DEAL_II_HOST_DEVICE void
    submit_divergence(const Number &div_in, const int q_point);

    /**
     * Submit the symmetric gradient @p sym_grad at quadrature point @p q_point
     * for subsequent integration via integrate() with
     * EvaluationFlags::gradients set. This function is only available when
     * the number of components equals the dimension (n_components_==dim).
     */
    DEAL_II_HOST_DEVICE void
    submit_symmetric_gradient(const SymmetricTensor<2, dim, Number> &sym_grad,
                              const int                              q_point);

  private:
    const unsigned int                                       dof_handler_index;
    const typename MatrixFree<dim, Number>::Data            *data;
    const typename MatrixFree<dim, Number>::PrecomputedData *precomputed_data;
    SharedData<dim, Number>                                 *shared_data;
    int                                                      cell_id;


    /**
     * Geometry classification of the current cell, loaded once at
     * construction so that the per-quadrature-point accessors do not read it
     * from global memory repeatedly.
     */
    ::dealii::internal::MatrixFreeFunctions::GeometryType cell_type;

    /**
     * Index of the first entry of the current cell in the compressed
     * inv_jacobian and JxW storage, loaded once at construction.
     */
    unsigned int mapping_data_offset;

    /**
     * For Cartesian and affine cells, the (constant) Jacobian determinant of
     * the current cell, loaded once at construction; JxW values are
     * reconstructed as cell_determinant * q_weights(q_point).
     */
    Number cell_determinant;

    /**
     * Return the index of the given quadrature point of the current cell in
     * the compressed inv_jacobian storage.
     */
    DEAL_II_HOST_DEVICE unsigned int
    inv_jacobian_index(const int q_point) const
    {
      return mapping_data_offset +
             ((cell_type == ::dealii::internal::MatrixFreeFunctions::general) ?
                q_point :
                0);
    }

    /**
     * Return the mapped quadrature weight (JxW) at the given quadrature
     * point of the current cell.
     */
    DEAL_II_HOST_DEVICE Number
    JxW_value(const int q_point) const
    {
      return (cell_type == ::dealii::internal::MatrixFreeFunctions::general) ?
               precomputed_data->JxW(mapping_data_offset + q_point) :
               cell_determinant * precomputed_data->q_weights(q_point);
    }
  };



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    FEEvaluation(const data_type *data, const unsigned int dof_handler_index)
    : dof_handler_index(dof_handler_index)
    , data(data)
    , precomputed_data(&data->precomputed_data[dof_handler_index])
    , shared_data(&data->shared_data[dof_handler_index])
    , cell_id(data->team_member.league_rank())
    , cell_type(precomputed_data->cell_type(cell_id))
    , mapping_data_offset(precomputed_data->data_index_offsets(cell_id))
    , cell_determinant(
        (cell_type != ::dealii::internal::MatrixFreeFunctions::general &&
         precomputed_data->JxW.size() > 0) ?
          precomputed_data->JxW(mapping_data_offset) :
          Number())
  {
    AssertIndexRange(dof_handler_index, data->n_dof_handler);

    Assert(
      n_components_ == precomputed_data->n_components,
      ExcMessage(
        "Portable::FEEvaluation initialized with wrong number of components. Should be " +
        Utilities::to_string(precomputed_data->n_components) +
        " but the template argument 4 is set to " +
        Utilities::to_string(n_components_)));

    // TODO: check fe_degree is correct by storing the used FE degree
    // inside PrecomputedData.
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE int
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_current_cell_index()
  {
    return precomputed_data->first_cell + cell_id;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE const typename FEEvaluation<dim,
                                                  fe_degree,
                                                  n_q_points_1d,
                                                  n_components_,
                                                  Number>::data_type *
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_matrix_free_data()
  {
    return data;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    read_dof_values(const DeviceVector<Number> &src)
  {
    // Populate the scratch memory
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(data->team_member, tensor_dofs_per_component),
      [&](const int &i) {
        for (unsigned int c = 0; c < n_components_; ++c)
          {
            const types::global_dof_index index =
              precomputed_data->local_to_global(i +
                                                  tensor_dofs_per_component * c,
                                                cell_id);
            // Dirichlet constrained DoFs read as zero
            shared_data->values(i, c) =
              (index == numbers::invalid_unsigned_int) ? Number(0) : src[index];
          }
      });
    data->team_member.team_barrier();

    for (unsigned int c = 0; c < n_components_; ++c)
      {
        if (precomputed_data->constraint_mask(cell_id * n_components + c) !=
            dealii::internal::MatrixFreeFunctions::ConstraintKinds::
              unconstrained)
          internal::resolve_hanging_nodes<dim, fe_degree, false, Number>(
            data->team_member,
            precomputed_data->constraint_weights,
            precomputed_data->constraint_mask(cell_id * n_components + c),
            Kokkos::subview(shared_data->values, Kokkos::ALL, c));
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    distribute_local_to_global(DeviceVector<Number> &dst) const
  {
    for (unsigned int c = 0; c < n_components_; ++c)
      {
        if (precomputed_data->constraint_mask(cell_id * n_components + c) !=
            dealii::internal::MatrixFreeFunctions::ConstraintKinds::
              unconstrained)
          internal::resolve_hanging_nodes<dim, fe_degree, true, Number>(
            data->team_member,
            precomputed_data->constraint_weights,
            precomputed_data->constraint_mask(cell_id * n_components + c),
            Kokkos::subview(shared_data->values, Kokkos::ALL, c));
      }

    if (precomputed_data->use_coloring)
      {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(data->team_member,
                                                     tensor_dofs_per_component),
                             [&](const int &i) {
                               for (unsigned int c = 0; c < n_components_; ++c)
                                 {
                                   const types::global_dof_index index =
                                     precomputed_data->local_to_global(
                                       i + tensor_dofs_per_component * c,
                                       cell_id);
                                   if (index != numbers::invalid_unsigned_int)
                                     dst[index] += shared_data->values(i, c);
                                 }
                             });
      }
    else
      {
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(data->team_member, tensor_dofs_per_component),
          [&](const int &i) {
            for (unsigned int c = 0; c < n_components_; ++c)
              {
                const types::global_dof_index index =
                  precomputed_data->local_to_global(
                    i + tensor_dofs_per_component * c, cell_id);
                if (index != numbers::invalid_unsigned_int)
                  Kokkos::atomic_add(&dst[index], shared_data->values(i, c));
              }
          });
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>::evaluate(
    const EvaluationFlags::EvaluationFlags evaluation_flag)
  {
    using ElementType = ::dealii::internal::MatrixFreeFunctions::ElementType;

    if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
        precomputed_data->element_type ==
          ElementType::tensor_symmetric_collocation)
      {
        internal::FEEvaluationImplCollocation<dim, fe_degree, Number>::evaluate(
          dof_handler_index, n_components, evaluation_flag, data);
      }
    // '<=' on type means tensor_symmetric or tensor_symmetric_hermite, see
    // shape_info.h for more details
    else if (fe_degree >= 0 &&
             internal::use_collocation_evaluation(fe_degree, n_q_points_1d) &&
             precomputed_data->element_type <= ElementType::tensor_symmetric)
      {
        internal::FEEvaluationImplTransformToCollocation<
          dim,
          fe_degree,
          n_q_points_1d,
          Number>::evaluate(dof_handler_index,
                            n_components,
                            evaluation_flag,
                            data);
      }
    else if (fe_degree >= 0 && precomputed_data->element_type <=
                                 ElementType::tensor_symmetric_no_collocation)
      {
        internal::FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::
          evaluate(dof_handler_index, n_components, evaluation_flag, data);
      }
    else
      {
        Kokkos::abort("The element type is not yet supported by the portable "
                      "matrix-free module.");
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::integrate(
    const EvaluationFlags::EvaluationFlags integration_flag)
  {
    using ElementType = ::dealii::internal::MatrixFreeFunctions::ElementType;

    if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
        precomputed_data->element_type ==
          ElementType::tensor_symmetric_collocation)
      {
        internal::FEEvaluationImplCollocation<dim, fe_degree, Number>::
          integrate(dof_handler_index, n_components, integration_flag, data);
      }
    // '<=' on type means tensor_symmetric or tensor_symmetric_hermite, see
    // shape_info.h for more details
    else if (fe_degree >= 0 &&
             internal::use_collocation_evaluation(fe_degree, n_q_points_1d) &&
             precomputed_data->element_type <= ElementType::tensor_symmetric)
      {
        internal::FEEvaluationImplTransformToCollocation<
          dim,
          fe_degree,
          n_q_points_1d,
          Number>::integrate(dof_handler_index,
                             n_components,
                             integration_flag,
                             data);
      }
    else if (fe_degree >= 0 && precomputed_data->element_type <=
                                 ElementType::tensor_symmetric_no_collocation)
      {
        internal::FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::
          integrate(dof_handler_index, n_components, integration_flag, data);
      }
    else
      {
        Kokkos::abort("The element type is not yet supported by the portable "
                      "matrix-free module.");
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components_,
                                            Number>::value_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_value(
    const int q_point) const
  {
    AssertIndexRange(q_point, n_q_points);
    if constexpr (n_components_ == 1)
      {
        return shared_data->values(q_point, 0);
      }
    else
      {
        value_type result;
        for (unsigned int c = 0; c < n_components; ++c)
          result[c] = shared_data->values(q_point, c);
        return result;
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components_,
                                            Number>::value_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_dof_value(const int dof_index) const
  {
    AssertIndexRange(dof_index, tensor_dofs_per_component);
    if constexpr (n_components_ == 1)
      {
        return shared_data->values(dof_index, 0);
      }
    else
      {
        value_type result;
        for (unsigned int c = 0; c < n_components; ++c)
          result[c] = shared_data->values(dof_index, c);
        return result;
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_value(const value_type &value, const int q_point)
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(precomputed_data->JxW.size() > 0,
           ExcMessage("submit_value() requires precomputed JxW"));
    const Number JxW = JxW_value(q_point);
    if constexpr (n_components_ == 1)
      {
        shared_data->values(q_point, 0) = value * JxW;
      }
    else
      {
        for (unsigned int c = 0; c < n_components; ++c)
          shared_data->values(q_point, c) = value[c] * JxW;
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_dof_value(const value_type &value, const int dof_index)
  {
    AssertIndexRange(dof_index, tensor_dofs_per_component);
    if constexpr (n_components_ == 1)
      {
        shared_data->values(dof_index, 0) = value;
      }
    else
      {
        for (unsigned int c = 0; c < n_components; ++c)
          shared_data->values(dof_index, c) = value[c];
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components_,
                                            Number>::gradient_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_gradient(const int q_point) const
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(precomputed_data->inv_jacobian.size() > 0,
           ExcMessage("get_gradient() requires precomputed inv_jacobian"));
    gradient_type grad;

    const unsigned int inv_jac_idx = inv_jacobian_index(q_point);
    if (cell_type == ::dealii::internal::MatrixFreeFunctions::cartesian)
      {
        if constexpr (n_components_ == 1)
          {
            for (unsigned int d = 0; d < dim; ++d)
              grad[d] = precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                        shared_data->gradients(q_point, d, 0);
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              for (unsigned int d = 0; d < dim; ++d)
                grad[c][d] = precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                             shared_data->gradients(q_point, d, c);
          }
      }
    else if constexpr (n_components_ == 1)
      {
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_2, d_1) *
                     shared_data->gradients(q_point, d_2, 0);
            grad[d_1] = tmp;
          }
      }
    else
      {
        for (unsigned int c = 0; c < n_components; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_2, d_1) *
                       shared_data->gradients(q_point, d_2, c);
              grad[c][d_1] = tmp;
            }
      }

    return grad;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_gradient(const gradient_type &gradient, const int q_point)
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(precomputed_data->inv_jacobian.size() > 0,
           ExcMessage("submit_gradient() requires precomputed inv_jacobian"));
    Assert(precomputed_data->JxW.size() > 0,
           ExcMessage("submit_gradient() requires precomputed JxW"));

    const unsigned int inv_jac_idx = inv_jacobian_index(q_point);
    const Number       JxW         = JxW_value(q_point);
    if (cell_type == ::dealii::internal::MatrixFreeFunctions::cartesian)
      {
        if constexpr (n_components_ == 1)
          {
            for (unsigned int d = 0; d < dim; ++d)
              shared_data->gradients(q_point, d, 0) =
                precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                gradient[d] * JxW;
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              for (unsigned int d = 0; d < dim; ++d)
                shared_data->gradients(q_point, d, c) =
                  precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                  gradient[c][d] * JxW;
          }
      }
    else if constexpr (n_components_ == 1)
      {
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                     gradient[d_2];
            shared_data->gradients(q_point, d_1, 0) = tmp * JxW;
          }
      }
    else
      {
        for (unsigned int c = 0; c < n_components; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                       gradient[c][d_2];
              shared_data->gradients(q_point, d_1, c) = tmp * JxW;
            }
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE SymmetricTensor<2, dim, Number>
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_symmetric_gradient(const int q_point) const
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(n_components_ == dim,
           ExcMessage("Function get_symmetric_gradient() only works when the "
                      "number of components and the number of dimensions are "
                      "equal."));

    return symmetrize(get_gradient(q_point));
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE Number
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_divergence(const int q_point) const
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(n_components_ == dim,
           ExcMessage("Function get_divergence() only works when the "
                      "number of components and the number of dimensions are "
                      "equal."));
    Assert(precomputed_data->inv_jacobian.size() > 0,
           ExcMessage("get_divergence() requires precomputed inv_jacobian"));

    const unsigned int inv_jac_idx = inv_jacobian_index(q_point);
    Number             divergence  = 0.;
    for (unsigned int c = 0; c < dim; ++c)
      for (unsigned int d = 0; d < dim; ++d)
        divergence += precomputed_data->inv_jacobian(inv_jac_idx, d, c) *
                      shared_data->gradients(q_point, d, c);
    return divergence;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_divergence(const Number &div_in, const int q_point)
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(n_components_ == dim,
           ExcMessage("Function submit_divergence() only works when the "
                      "number of components and the number of dimensions are "
                      "equal."));
    Assert(precomputed_data->inv_jacobian.size() > 0,
           ExcMessage("submit_divergence() requires precomputed inv_jacobian"));
    Assert(precomputed_data->JxW.size() > 0,
           ExcMessage("submit_divergence() requires precomputed JxW"));

    const unsigned int inv_jac_idx = inv_jacobian_index(q_point);
    const Number       JxW         = JxW_value(q_point);
    for (unsigned int c = 0; c < dim; ++c)
      for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
        shared_data->gradients(q_point, d_1, c) =
          precomputed_data->inv_jacobian(inv_jac_idx, d_1, c) * div_in * JxW;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  DEAL_II_HOST_DEVICE void
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_symmetric_gradient(const SymmetricTensor<2, dim, Number> &sym_grad,
                              const int                              q_point)
  {
    AssertIndexRange(q_point, n_q_points);
    Assert(n_components_ == dim,
           ExcMessage("Function submit_symmetric_gradient() only works when "
                      "the number of components and the number of dimensions "
                      "are equal."));
    Assert(precomputed_data->inv_jacobian.size() > 0,
           ExcMessage(
             "submit_symmetric_gradient() requires precomputed inv_jacobian"));
    Assert(precomputed_data->JxW.size() > 0,
           ExcMessage("submit_symmetric_gradient() requires precomputed JxW"));

    const unsigned int inv_jac_idx = inv_jacobian_index(q_point);
    const Number       JxW         = JxW_value(q_point);
    for (unsigned int c = 0; c < dim; ++c)
      for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
        {
          Number tmp = 0.;
          for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
            tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                   sym_grad[c][d_2];
          shared_data->gradients(q_point, d_1, c) = tmp * JxW;
        }
  }



#ifndef DOXYGEN
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  constexpr unsigned int
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
      n_q_points;
#endif



  namespace internal
  {
    namespace batched
    {
      /**
       * Same as Portable::FEEvaluation, but for all cells of
       * BatchedData::cell_range at once. The index @p q_point of the
       * functions below runs over the quadrature points of all cells of the
       * batch stored cell after cell, as passed to the functor of
       * BatchedData::for_each_quad_point(), and @p dof_index over their
       * degrees of freedom.
       */
      template <int dim,
                int fe_degree,
                int n_q_points_1d = fe_degree + 1,
                int n_components_ = 1,
                typename Number   = double>
      class FEEvaluation
      {
      public:
        /**
         * An alias for the value type. This is @p Number for scalar problems
         * and Tensor<1, n_components> for vector-valued problems.
         */
        using value_type = std::conditional_t<(n_components_ == 1),
                                              Number,
                                              Tensor<1, n_components_, Number>>;

        /**
         * An alias for the gradient type.
         */
        using gradient_type = std::conditional_t<
          n_components_ == 1,
          Tensor<1, dim, Number>,
          std::conditional_t<n_components_ == dim,
                             Tensor<2, dim, Number>,
                             Tensor<1, n_components_, Tensor<1, dim, Number>>>>;

        /**
         * An alias to kernel specific information.
         */
        using data_type = BatchedData<dim, Number>;

        /**
         * Dimension.
         */
        static constexpr unsigned int dimension = dim;

        /**
         * Number of components.
         */
        static constexpr unsigned int n_components = n_components_;

        /**
         * Number of quadrature points per cell.
         */
        static constexpr unsigned int n_q_points_per_cell =
          Utilities::pow(n_q_points_1d, dim);

        /**
         * Number of tensor degrees of freedom of a scalar component per cell.
         */
        static constexpr unsigned int tensor_dofs_per_component =
          Utilities::pow(fe_degree + 1, dim);

        /**
         * Number of tensor degrees of freedom of all components per cell.
         */
        static constexpr unsigned int tensor_dofs_per_cell =
          tensor_dofs_per_component * n_components;

        /**
         * Constructor, see Portable::FEEvaluation.
         */
        DEAL_II_HOST_DEVICE
        explicit FEEvaluation(const data_type   *data,
                              const unsigned int dof_handler_index = 0);

        /**
         * Return the index of the cell the quadrature point @p q_point
         * belongs to.
         */
        DEAL_II_HOST_DEVICE
        int
        get_current_cell_index(const int q_point) const;

        /**
         * Return a pointer to the BatchedData object.
         */
        DEAL_II_HOST_DEVICE
        const data_type *
        get_matrix_free_data();

        /**
         * Read the values of @p src on the degrees of freedom of the cells of
         * the batch.
         */
        DEAL_II_HOST_DEVICE void
        read_dof_values(const DeviceVector<Number> &src);

        /**
         * Sum the values stored on the degrees of freedom of the cells of the
         * batch into @p dst. The function ends with a team barrier, so that
         * the team can go on with the next batch.
         */
        DEAL_II_HOST_DEVICE void
        distribute_local_to_global(DeviceVector<Number> &dst) const;

        /**
         * Evaluate the values and gradients at the quadrature points as
         * specified by @p evaluate_flag.
         */
        DEAL_II_HOST_DEVICE void
        evaluate(const EvaluationFlags::EvaluationFlags evaluate_flag);

        /**
         * Test the submitted values and gradients by the basis functions as
         * specified by @p integration_flag.
         */
        DEAL_II_HOST_DEVICE void
        integrate(const EvaluationFlags::EvaluationFlags integration_flag);

        /**
         * Same as Portable::FEEvaluation::get_value().
         */
        DEAL_II_HOST_DEVICE value_type
        get_value(const int q_point) const;

        /**
         * Same as Portable::FEEvaluation::get_dof_value().
         */
        DEAL_II_HOST_DEVICE value_type
        get_dof_value(const int dof_index) const;

        /**
         * Same as Portable::FEEvaluation::submit_value().
         */
        DEAL_II_HOST_DEVICE void
        submit_value(const value_type &val_in, const int q_point);

        /**
         * Same as Portable::FEEvaluation::submit_dof_value().
         */
        DEAL_II_HOST_DEVICE void
        submit_dof_value(const value_type &value, const int dof_index);

        /**
         * Same as Portable::FEEvaluation::get_gradient().
         */
        DEAL_II_HOST_DEVICE gradient_type
        get_gradient(const int q_point) const;

        /**
         * Same as Portable::FEEvaluation::submit_gradient().
         */
        DEAL_II_HOST_DEVICE void
        submit_gradient(const gradient_type &gradient, const int q_point);

        /**
         * Same as Portable::FEEvaluation::get_symmetric_gradient().
         */
        DEAL_II_HOST_DEVICE
        SymmetricTensor<2, dim, Number>
        get_symmetric_gradient(const int q_point) const;

        /**
         * Same as Portable::FEEvaluation::get_divergence().
         */
        DEAL_II_HOST_DEVICE Number
        get_divergence(const int q_point) const;

        /**
         * Same as Portable::FEEvaluation::submit_divergence().
         */
        DEAL_II_HOST_DEVICE void
        submit_divergence(const Number &div_in, const int q_point);

        /**
         * Same as Portable::FEEvaluation::submit_symmetric_gradient().
         */
        DEAL_II_HOST_DEVICE void
        submit_symmetric_gradient(
          const SymmetricTensor<2, dim, Number> &sym_grad,
          const int                              q_point);

      private:
        const unsigned int dof_handler_index;
        const data_type   *data;
        const typename MatrixFree<dim, Number>::PrecomputedData
                                *precomputed_data;
        SharedData<dim, Number> *shared_data;

        /**
         * Index of the first cell of the batch, BatchedData::cell_range.first.
         */
        const int first_cell_id;

        /**
         * Number of cells in the batch.
         */
        const int batch_size;

        /**
         * Number of quadrature points of all cells in the batch.
         */
        const int n_q_points_per_cell_range;

        /**
         * Whether any cell of the batch has hanging node constraints.
         */
        const bool has_hanging_nodes;

        /**
         * Index within the current color of the cell the quadrature point
         * @p q_point belongs to.
         */
        DEAL_II_HOST_DEVICE int
        cell_id(const int q_point) const
        {
          return first_cell_id +
                 q_point / static_cast<int>(n_q_points_per_cell);
        }

        /**
         * Index of the quadrature point @p q_point within its cell.
         */
        DEAL_II_HOST_DEVICE int
        cell_q_point(const int q_point) const
        {
          return q_point % static_cast<int>(n_q_points_per_cell);
        }

        /**
         * Check whether any cell of the batch has hanging node constraints.
         */
        DEAL_II_HOST_DEVICE bool
        batch_has_hanging_nodes() const
        {
          using ConstraintKinds =
            dealii::internal::MatrixFreeFunctions::ConstraintKinds;

          ConstraintKinds batch_mask = ConstraintKinds::unconstrained;
          for (int i = 0; i < batch_size * n_components_; ++i)
            batch_mask |= precomputed_data->constraint_mask(
              first_cell_id * n_components_ + i);
          return batch_mask != ConstraintKinds::unconstrained;
        }

        /**
         * Copy the hanging node constraint weights to the scratch pad, which
         * is only in use during evaluate() and integrate(), and return a view
         * of them. The caller has to add a team barrier.
         */
        DEAL_II_HOST_DEVICE
        typename SharedData<dim, Number>::SharedViewScratchPad
        stage_constraint_weights() const
        {
          const int n_weights = precomputed_data->constraint_weights.size();
          Assert(shared_data->scratch_pad.size() >=
                   static_cast<std::size_t>(n_weights),
                 ExcInternalError());

          const typename SharedData<dim, Number>::SharedViewScratchPad weights =
            Kokkos::subview(shared_data->scratch_pad,
                            Kokkos::make_pair(0, n_weights));
          Kokkos::parallel_for(Kokkos::TeamVectorRange(data->team_member,
                                                       n_weights),
                               [&](const int &i) {
                                 weights(i) =
                                   precomputed_data->constraint_weights(i);
                               });
          return weights;
        }

        /**
         * The hanging node constraint masks of component @p component of the
         * cells of the batch.
         */
        DEAL_II_HOST_DEVICE auto
        constraint_masks(const unsigned int component) const
        {
          // stored as [cell * n_components + component]
          const Kokkos::View<
            const dealii::internal::MatrixFreeFunctions::ConstraintKinds **,
            Kokkos::LayoutRight,
            MemorySpace::Default::kokkos_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            masks(precomputed_data->constraint_mask.data(),
                  precomputed_data->n_cells,
                  n_components_);
          return Kokkos::subview(masks,
                                 Kokkos::make_pair(first_cell_id,
                                                   first_cell_id + batch_size),
                                 component);
        }
      };



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        FEEvaluation(const data_type   *data,
                     const unsigned int dof_handler_index)
        : dof_handler_index(dof_handler_index)
        , data(data)
        , precomputed_data(&data->precomputed_data[dof_handler_index])
        , shared_data(&data->shared_data[dof_handler_index])
        , first_cell_id(data->cell_range.first)
        , batch_size(data->batch_size())
        , n_q_points_per_cell_range(batch_size * n_q_points_per_cell)
        , has_hanging_nodes(batch_has_hanging_nodes())
      {
        AssertIndexRange(dof_handler_index, data->n_dof_handler);
        Assert(n_components_ == precomputed_data->n_components,
               ExcMessage("Wrong number of components."));
        Assert(first_cell_id >= 0, ExcInternalError());
        AssertIndexRange(first_cell_id + batch_size,
                         precomputed_data->n_cells + 1);
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE int
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_current_cell_index(const int q_point) const
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        return precomputed_data->first_cell + cell_id(q_point);
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE const typename FEEvaluation<dim,
                                                      fe_degree,
                                                      n_q_points_1d,
                                                      n_components_,
                                                      Number>::data_type *
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_matrix_free_data()
      {
        return data;
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        read_dof_values(const DeviceVector<Number> &src)
      {
        constexpr int n_dofs = tensor_dofs_per_component;

        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(data->team_member, batch_size * n_dofs),
          [&](const int &i) {
            const int cell = first_cell_id + i / n_dofs;
            const int dof  = i % n_dofs;
            for (unsigned int c = 0; c < n_components_; ++c)
              {
                const types::global_dof_index index =
                  precomputed_data->local_to_global(dof + n_dofs * c, cell);
                // Dirichlet constrained DoFs read as zero
                shared_data->values(i, c) =
                  (index == numbers::invalid_unsigned_int) ? Number(0) :
                                                             src[index];
              }
          });
        typename SharedData<dim, Number>::SharedViewScratchPad
          constraint_weights;
        if (has_hanging_nodes)
          constraint_weights = stage_constraint_weights();
        data->team_member.team_barrier();

        if (has_hanging_nodes)
          for (unsigned int c = 0; c < n_components_; ++c)
            resolve_hanging_nodes<dim, fe_degree, false>(
              data->team_member,
              constraint_weights,
              constraint_masks(c),
              Kokkos::subview(shared_data->values, Kokkos::ALL, c),
              batch_size);
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        distribute_local_to_global(DeviceVector<Number> &dst) const
      {
        constexpr int n_dofs = tensor_dofs_per_component;

        if (has_hanging_nodes)
          {
            const auto constraint_weights = stage_constraint_weights();
            data->team_member.team_barrier();

            for (unsigned int c = 0; c < n_components_; ++c)
              resolve_hanging_nodes<dim, fe_degree, true>(
                data->team_member,
                constraint_weights,
                constraint_masks(c),
                Kokkos::subview(shared_data->values, Kokkos::ALL, c),
                batch_size);
          }

        if (precomputed_data->use_coloring)
          {
            Kokkos::parallel_for(
              Kokkos::TeamVectorRange(data->team_member, batch_size * n_dofs),
              [&](const int &i) {
                const int cell = first_cell_id + i / n_dofs;
                const int dof  = i % n_dofs;
                for (unsigned int c = 0; c < n_components_; ++c)
                  {
                    const types::global_dof_index index =
                      precomputed_data->local_to_global(dof + n_dofs * c, cell);
                    if (index != numbers::invalid_unsigned_int)
                      dst[index] += shared_data->values(i, c);
                  }
              });
          }
        else
          {
            Kokkos::parallel_for(
              Kokkos::TeamVectorRange(data->team_member, batch_size * n_dofs),
              [&](const int &i) {
                const int cell = first_cell_id + i / n_dofs;
                const int dof  = i % n_dofs;
                for (unsigned int c = 0; c < n_components_; ++c)
                  {
                    const types::global_dof_index index =
                      precomputed_data->local_to_global(dof + n_dofs * c, cell);
                    if (index != numbers::invalid_unsigned_int)
                      Kokkos::atomic_add(&dst[index],
                                         shared_data->values(i, c));
                  }
              });
          }

        // the next batch of the team reuses the shared memory
        data->team_member.team_barrier();
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag)
      {
        using ElementType =
          ::dealii::internal::MatrixFreeFunctions::ElementType;

        if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
            precomputed_data->element_type ==
              ElementType::tensor_symmetric_collocation)
          {
            FEEvaluationImplCollocation<dim, fe_degree, Number>::evaluate(
              dof_handler_index, n_components, evaluation_flag, data);
          }
        // '<=' on type means tensor_symmetric or tensor_symmetric_hermite, see
        // shape_info.h for more details
        else if (fe_degree >= 0 &&
                 use_collocation_evaluation(fe_degree, n_q_points_1d) &&
                 precomputed_data->element_type <=
                   ElementType::tensor_symmetric)
          {
            FEEvaluationImplTransformToCollocation<
              dim,
              fe_degree,
              n_q_points_1d,
              Number>::evaluate(dof_handler_index,
                                n_components,
                                evaluation_flag,
                                data);
          }
        else if (fe_degree >= 0 &&
                 precomputed_data->element_type <=
                   ElementType::tensor_symmetric_no_collocation)
          {
            FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::evaluate(
              dof_handler_index, n_components, evaluation_flag, data);
          }
        else
          {
            Kokkos::abort(
              "The element type is not yet supported by the portable "
              "matrix-free module.");
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        integrate(const EvaluationFlags::EvaluationFlags integration_flag)
      {
        using ElementType =
          ::dealii::internal::MatrixFreeFunctions::ElementType;

        if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
            precomputed_data->element_type ==
              ElementType::tensor_symmetric_collocation)
          {
            FEEvaluationImplCollocation<dim, fe_degree, Number>::integrate(
              dof_handler_index, n_components, integration_flag, data);
          }
        // '<=' on type means tensor_symmetric or tensor_symmetric_hermite, see
        // shape_info.h for more details
        else if (fe_degree >= 0 &&
                 use_collocation_evaluation(fe_degree, n_q_points_1d) &&
                 precomputed_data->element_type <=
                   ElementType::tensor_symmetric)
          {
            FEEvaluationImplTransformToCollocation<
              dim,
              fe_degree,
              n_q_points_1d,
              Number>::integrate(dof_handler_index,
                                 n_components,
                                 integration_flag,
                                 data);
          }
        else if (fe_degree >= 0 &&
                 precomputed_data->element_type <=
                   ElementType::tensor_symmetric_no_collocation)
          {
            FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::integrate(
              dof_handler_index, n_components, integration_flag, data);
          }
        else
          {
            Kokkos::abort(
              "The element type is not yet supported by the portable "
              "matrix-free module.");
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                                fe_degree,
                                                n_q_points_1d,
                                                n_components_,
                                                Number>::value_type
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_value(const int q_point) const
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        if constexpr (n_components_ == 1)
          {
            return shared_data->values(q_point, 0);
          }
        else
          {
            value_type result;
            for (unsigned int c = 0; c < n_components; ++c)
              result[c] = shared_data->values(q_point, c);
            return result;
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                                fe_degree,
                                                n_q_points_1d,
                                                n_components_,
                                                Number>::value_type
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_dof_value(const int dof_index) const
      {
        AssertIndexRange(dof_index, batch_size * tensor_dofs_per_component);
        if constexpr (n_components_ == 1)
          {
            return shared_data->values(dof_index, 0);
          }
        else
          {
            value_type result;
            for (unsigned int c = 0; c < n_components; ++c)
              result[c] = shared_data->values(dof_index, c);
            return result;
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        submit_value(const value_type &value, const int q_point)
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        Assert(precomputed_data->JxW.size() > 0,
               ExcMessage("submit_value() requires precomputed JxW"));
        const Number JxW =
          precomputed_data->JxW_value(cell_id(q_point), cell_q_point(q_point));
        if constexpr (n_components_ == 1)
          {
            shared_data->values(q_point, 0) = value * JxW;
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              shared_data->values(q_point, c) = value[c] * JxW;
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        submit_dof_value(const value_type &value, const int dof_index)
      {
        AssertIndexRange(dof_index, batch_size * tensor_dofs_per_component);
        if constexpr (n_components_ == 1)
          {
            shared_data->values(dof_index, 0) = value;
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              shared_data->values(dof_index, c) = value[c];
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                                fe_degree,
                                                n_q_points_1d,
                                                n_components_,
                                                Number>::gradient_type
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_gradient(const int q_point) const
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        Assert(precomputed_data->inv_jacobian.size() > 0,
               ExcMessage("get_gradient() requires precomputed inv_jacobian"));
        gradient_type grad;

        const int          cell = cell_id(q_point);
        const unsigned int inv_jac_idx =
          precomputed_data->inv_jacobian_index(cell, cell_q_point(q_point));
        if (precomputed_data->cell_type(cell) ==
            ::dealii::internal::MatrixFreeFunctions::cartesian)
          {
            if constexpr (n_components_ == 1)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  grad[d] = precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                            shared_data->gradients(q_point, d, 0);
              }
            else
              {
                for (unsigned int c = 0; c < n_components; ++c)
                  for (unsigned int d = 0; d < dim; ++d)
                    grad[c][d] =
                      precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                      shared_data->gradients(q_point, d, c);
              }
          }
        else if constexpr (n_components_ == 1)
          {
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_2, d_1) *
                         shared_data->gradients(q_point, d_2, 0);
                grad[d_1] = tmp;
              }
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                    tmp +=
                      precomputed_data->inv_jacobian(inv_jac_idx, d_2, d_1) *
                      shared_data->gradients(q_point, d_2, c);
                  grad[c][d_1] = tmp;
                }
          }

        return grad;
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        submit_gradient(const gradient_type &gradient, const int q_point)
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        Assert(precomputed_data->inv_jacobian.size() > 0,
               ExcMessage(
                 "submit_gradient() requires precomputed inv_jacobian"));
        Assert(precomputed_data->JxW.size() > 0,
               ExcMessage("submit_gradient() requires precomputed JxW"));

        const int          cell = cell_id(q_point);
        const int          q    = cell_q_point(q_point);
        const unsigned int inv_jac_idx =
          precomputed_data->inv_jacobian_index(cell, q);
        const Number JxW = precomputed_data->JxW_value(cell, q);
        if (precomputed_data->cell_type(cell) ==
            ::dealii::internal::MatrixFreeFunctions::cartesian)
          {
            if constexpr (n_components_ == 1)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  shared_data->gradients(q_point, d, 0) =
                    precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                    gradient[d] * JxW;
              }
            else
              {
                for (unsigned int c = 0; c < n_components; ++c)
                  for (unsigned int d = 0; d < dim; ++d)
                    shared_data->gradients(q_point, d, c) =
                      precomputed_data->inv_jacobian(inv_jac_idx, d, d) *
                      gradient[c][d] * JxW;
              }
          }
        else if constexpr (n_components_ == 1)
          {
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                         gradient[d_2];
                shared_data->gradients(q_point, d_1, 0) = tmp * JxW;
              }
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                    tmp +=
                      precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                      gradient[c][d_2];
                  shared_data->gradients(q_point, d_1, c) = tmp * JxW;
                }
          }
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE SymmetricTensor<2, dim, Number>
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_symmetric_gradient(const int q_point) const
      {
        Assert(n_components_ == dim,
               ExcMessage(
                 "Function get_symmetric_gradient() only works when the "
                 "number of components and the number of dimensions are "
                 "equal."));

        return symmetrize(get_gradient(q_point));
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE Number
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        get_divergence(const int q_point) const
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        Assert(n_components_ == dim,
               ExcMessage("Function get_divergence() only works when the "
                          "number of components and the number of dimensions "
                          "are equal."));
        Assert(precomputed_data->inv_jacobian.size() > 0,
               ExcMessage(
                 "get_divergence() requires precomputed inv_jacobian"));

        const unsigned int inv_jac_idx =
          precomputed_data->inv_jacobian_index(cell_id(q_point),
                                               cell_q_point(q_point));
        Number divergence = 0.;
        for (unsigned int c = 0; c < dim; ++c)
          for (unsigned int d = 0; d < dim; ++d)
            divergence += precomputed_data->inv_jacobian(inv_jac_idx, d, c) *
                          shared_data->gradients(q_point, d, c);
        return divergence;
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        submit_divergence(const Number &div_in, const int q_point)
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        Assert(n_components_ == dim,
               ExcMessage("Function submit_divergence() only works when the "
                          "number of components and the number of dimensions "
                          "are equal."));
        Assert(precomputed_data->inv_jacobian.size() > 0,
               ExcMessage(
                 "submit_divergence() requires precomputed inv_jacobian"));
        Assert(precomputed_data->JxW.size() > 0,
               ExcMessage("submit_divergence() requires precomputed JxW"));

        const int          cell = cell_id(q_point);
        const int          q    = cell_q_point(q_point);
        const unsigned int inv_jac_idx =
          precomputed_data->inv_jacobian_index(cell, q);
        const Number JxW = precomputed_data->JxW_value(cell, q);
        for (unsigned int c = 0; c < dim; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            shared_data->gradients(q_point, d_1, c) =
              precomputed_data->inv_jacobian(inv_jac_idx, d_1, c) * div_in *
              JxW;
      }



      template <int dim,
                int fe_degree,
                int n_q_points_1d,
                int n_components_,
                typename Number>
      DEAL_II_HOST_DEVICE void
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
        submit_symmetric_gradient(
          const SymmetricTensor<2, dim, Number> &sym_grad,
          const int                              q_point)
      {
        AssertIndexRange(q_point, n_q_points_per_cell_range);
        Assert(n_components_ == dim,
               ExcMessage(
                 "Function submit_symmetric_gradient() only works when "
                 "the number of components and the number of dimensions "
                 "are equal."));
        Assert(precomputed_data->inv_jacobian.size() > 0,
               ExcMessage("submit_symmetric_gradient() requires precomputed "
                          "inv_jacobian"));
        Assert(precomputed_data->JxW.size() > 0,
               ExcMessage(
                 "submit_symmetric_gradient() requires precomputed JxW"));

        const int          cell = cell_id(q_point);
        const int          q    = cell_q_point(q_point);
        const unsigned int inv_jac_idx =
          precomputed_data->inv_jacobian_index(cell, q);
        const Number JxW = precomputed_data->JxW_value(cell, q);
        for (unsigned int c = 0; c < dim; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                       sym_grad[c][d_2];
              shared_data->gradients(q_point, d_1, c) = tmp * JxW;
            }
      }
    } // namespace batched
  }   // namespace internal
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
