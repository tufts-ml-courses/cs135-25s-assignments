# Yes, students should edit the TODO lines of this file.

import numpy as np
from scipy.special import logsumexp
from scipy.special import expit as logistic_sigmoid

# No other imports allowed.


def insert_final_col_of_all_ones(x_NF):
    ''' Append a column of all ones to provided array.

    Args
    ----
    x_NF : 2D array, size N x F

    Returns
    -------
    xone_NG : 2D array, size N x G, where G = F+1
        First F columns will be same as input array x_NF
        Final column will be equal to all ones.
    '''
    N = x_NF.shape[0]
    xone_NG = np.hstack([x_NF, np.ones((N, 1))])
    return xone_NG


class LogisticRegressionGradientDescent():
    ''' Logistic Regression binary classifier trainable via gradient descent.

    Object that implements the standard sklearn binary classifier API:
    * fit : train the model and set internal trainable attributes
    * predict : produces hard binary predictions
    * predict_proba : makes probabilistic predictions for labels 0 and 1

    Attributes set by calling __init__()
    ------------------------------------
    C : float
    step_size : float
    num_iterations : int

    Attributes set only by calling fit()
    ------------------------------------
    wtil_G : 1D array, size G = F + 1
        learned parameter array, stack weights for F features plus bias coef.
    trace_steps : list
    trace_loss : list
    trace_grad_L1_norm : list

    Training Objective
    ------------------
    In math, the loss is defined as:
        J(w,b) = \frac{1}{N} (C \sum_n scoreBCE(w, b, x_n, y_n) + 0.5 w^T w)
    Note the L2 penalty ONLY affects the weights in w, not bias coefficient.

    We can directly interpret J(w) as an upper bound on the error rate
    on the training data, because:
    * BCE is an upperbound on zero-one loss when done in base 2, as we do here
    * the extra L2 penalty term will only ever add to the loss

    Example Usage
    -------------
    >>> x_N1 = np.hstack([
    ...     np.linspace(-2, -1, 3),
    ...     np.linspace(+1, +2, 3)])[:,np.newaxis]
    >>> y_N = np.hstack([np.zeros(3), np.ones(3)])

    >>> clf = LogisticRegressionGradientDescent(
    ...     C=10.0, step_size=1.0, verbose=False)

    ### Shouldn't have any weights if we haven't trained yet
    >>> assert not hasattr(clf, 'wtil_G')

    ### After training, should have some weights
    >>> clf.fit(x_N1, y_N)
    >>> assert hasattr(clf, 'wtil_G')

    ### After training, should have converged
    >>> clf.did_converge
    True

    ### Show the positive-class probability
    >>> proba1_N = clf.predict_proba(x_N1)[:,1]
    >>> print(["%.2f" % phat for phat in proba1_N])
    ['0.00', '0.02', '0.06', '0.94', '0.98', '1.00']

    ### Show the hard binary predictions
    >>> clf.predict(x_N1).tolist()
    [0, 0, 0, 1, 1, 1]
    '''

    def __init__(
            self,
            C=1.0,
            step_size=0.0001,
            num_iterations=999,
            use_base2_for_BCE=True,
            init_recipe='zeros',
            random_state=0,
            verbose=True,
            loss_converge_thr=0.00001,
            grad_norm_converge_thr=0.001,
            param_converge_thr=0.001,
            proba_to_binary_threshold=0.5,
    ):
        ''' Construct instance and set its attributes

        Args
        ----
        C : float
        step_size : float
        num_iterations : int
        use_base2_for_BCE : bool
            If True, use log base 2 for the BCE loss, otherwise use base e.
        init_recipe : str
            Describes how wtil_G array is filled initially before any learning.
        random_state : int
            Controls pseudorandom generator that may be needed
            to fill the parameter array wtil_G initially.
        verbose : bool
            Print out progress updates every so often during gradient descent.

        Returns
        -------
        New instance of this class
        '''
        self.C = float(C)
        self.step_size = float(step_size)
        self.num_iterations = int(num_iterations)
        self.use_base2_for_BCE = bool(use_base2_for_BCE)
        self.init_recipe = str(init_recipe)
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.verbose = bool(verbose)
        self.loss_converge_thr = float(loss_converge_thr)
        self.grad_norm_converge_thr = float(grad_norm_converge_thr)
        self.param_converge_thr = float(param_converge_thr)
        self.proba_to_binary_threshold = float(proba_to_binary_threshold)

    # Prediction API methods

    def predict_proba(self, x_NF):
        ''' Produce soft probabilistic predictions for provided input features

        Args
        ----
        x_NF : 2D array, size N x F (n_examples x n_features_excluding_bias)
            Input feature vector (one row per example).

        Returns
        -------
        yproba_N2 : 2D array, size N x 2
            First column gives probability of zero label (negative)
            Second column gives probability of one label (positive)
            Each entry is a non-negative probability value within (0.0, 1.0)
            Each row sums to one
        '''
        if not hasattr(self, 'wtil_G'):
            raise AttributeError(
                "This LogisticRegression instance is not fitted yet."
                + " Must call 'fit' before 'predict_proba'."
                + " Set args carefully to ensure fit did not diverge.")
        N = x_NF.shape[0]
        w_F = None  # TODO unpack weight coefs from internal attribute wtil_G
        bias = -1.0  # TODO unpack bias coef from internal attribute wtil_G
        z_N = np.zeros(N)  # TODO compute weights times x plus b
        proba1_N = 1.0 * z_N  # TODO use 'logistic_sigmoid' to make probas
        proba0_N = None      # TODO convert probas of y=1 to probas of y=0

        proba_N2 = 0.5 + np.zeros((N, 2))  # TODO returnval fulfill docstring
        return proba_N2

    def predict(self, x_NF):
        ''' Produce hard binary predictions for provided input features

        Args
        ----
        x_NF : 2D array, size N x F (n_examples x n_features_excluding_bias)
            Input feature vectors (one row per example).

        Returns
        -------
        yhat_N : 1D array, size N, dtype int32
            Each entry is a binary integer value (either 0 or 1)
        '''
        if not hasattr(self, 'wtil_G'):
            raise AttributeError(
                "This LogisticRegression instance is not fitted yet."
                + " Must call 'fit' before 'predict'."
                + " Set args carefully to ensure fit did not diverge.")
        thr = self.proba_to_binary_threshold  # Unpack threshold
        proba_N2 = self.predict_proba(x_NF)
        yhat_bool_N = proba_N2.min(axis=1) < thr  # TODO fixme
        yhat_int_N = np.asarray(yhat_bool_N, dtype=np.int32)
        return yhat_int_N

    # Method for training

    def fit(self, x_NF, y_N):
        ''' Fit logistic regression model to provided training data

        Will minimize the loss function defined by calc_loss

        Returns
        -------
        Nothing. Only internal instance attributes updated.

        Post Condition
        --------------
        Internal attributes are updated:
        * wtil_G contains the optimal weights
        * trace_loss contains loss at every step of gradient descent
        * trace_grad_L1_norm contains L1 norm of grad after every step
        '''
        self.did_diverge = False
        self.did_converge = False
        self.trace_steps = list()
        self.trace_loss = list()
        self.trace_grad_L1_norm = list()
        self.trace_param = list()

        # Setup dimension attributes
        # F : num features excluding bias
        # G : num features including bias
        self.num_features_excluding_bias = x_NF.shape[1]
        self.F = x_NF.shape[1]
        self.G = self.F + 1

        # Setup input features with additional 'all ones' column
        xone_NG = insert_final_col_of_all_ones(x_NF)

        # Initialize wtil_G according to the selected recipe
        if self.verbose:
            print("Initializing G=%d parameters in wtil_G via recipe: %s" % (
                self.G, self.init_recipe))
        wtil_G = self.initialize_param_arr(xone_NG, y_N)

        # Run gradient descent!
        # Loop over iterations 0, 1, ..., num_iterations -1, num_iterations
        # We don't do a parameter update on iteration 0, just use the initial w
        if self.verbose:
            print("Running GD for up to %d iters with step_size %.3g" % (
                self.num_iterations, self.step_size))
        for iter_id in range(self.num_iterations + 1):
            if iter_id > 0:
                # TODO update parameter using self's 'step_size' attribute
                wtil_G = wtil_G - 0  # <- TODO replace this line

            loss = 4.321                      # TODO call calc_loss
            grad_G = 1.234 + np.zeros(self.G)  # TODO call calc_grad
            avg_L1_norm_of_grad = np.mean(np.abs(grad_G))

            # Print information to stdout
            if self.verbose:
                if iter_id < 20 or (iter_id % 20 == 0) or (iter_id % 20 == 1):
                    lineA = 'iter %3d/%d  loss % 10.6f' % (
                        iter_id, self.num_iterations, loss)
                    lineB = ' avg_abs_grad % 10.6f  w % 7.3f b % 7.3f' % (
                        avg_L1_norm_of_grad, wtil_G[0], wtil_G[-1])
                    print(lineA, lineB)

            # Record information
            self.trace_steps.append(iter_id)
            self.trace_loss.append(loss)
            self.trace_grad_L1_norm.append(avg_L1_norm_of_grad)
            self.trace_param.append(wtil_G)

            # Assess divergence and raise ValueError as soon as it happens
            self.raise_error_if_diverging(
                self.trace_steps, self.trace_loss, self.trace_grad_L1_norm)

            # Assess convergence and break early if happens
            self.did_converge = self.check_convergence(
                self.trace_steps, self.trace_loss,
                self.trace_grad_L1_norm, self.trace_param)
            if self.did_converge:
                break

        assert not self.did_diverge
        self.wtil_G = wtil_G
        if self.did_converge:
            self.solver_status = (
                "Done. Converged after %d iters with step_size %.5g" % (
                    self.trace_steps[-1], self.step_size))
        else:
            self.solver_status = (
                "Done. NOT converged after %d iters with step_size %.5g" % (
                    self.trace_steps[-1], self.step_size))

        if self.verbose:
            print(self.solver_status)
        # Done with `fit`.

    # Methods for gradient descent: calc_loss and calc_grad

    def calc_loss(self, wtil_G, xone_NG, y_N):
        ''' Compute total loss for used to train logistic regression.

        Sum of log loss over training examples plus L2 penalty term.

        Args
        ----
        wtil_G : 1D array, size G 
            Combined vector of weights and bias
        xone_NG : 2D array, size N x G (n_examples x n_features+1)
            Input features, with last column of all ones
        y_N : 1D array, size N
            Binary labels for each example (either 0 or 1)

        Returns
        -------
        loss : float
            Scalar loss. Lower is better.
        '''
        N = float(y_N.size)
        G = wtil_G.size
        C = float(self.C)

        # First term: Calc loss due to L2 penalty on weights
        L2_loss = 0.0  # TODO fixme. Remember to omit the bias coef from this.

        # Second term: Calc scoreBCE summed over all examples in dataset
        sumBCE_loss = 0.0  # TODO fixme. Use base e.

        # Conversion to base 2. Done for you.
        if self.use_base2_for_BCE:
            sumBCE_loss = sumBCE_loss / np.log(2)

        # Add two terms together and rescale. Done for you.
        return (L2_loss + C * sumBCE_loss) / (N * C)

    def calc_grad(self, wtil_G, xone_NG, y_N):
        ''' Compute gradient of total loss for training logistic regression.

        Args
        ----
        wtil_G : 1D array, size G (G = n_features_including_bias)
            Combined vector of weights and bias
        xone_NG : 2D array, size N x G (n_examples x n_features_including_bias)
            Input features, with last column of all ones
        y_N : 1D array, size N
            Binary labels for each example (either 0 or 1)

        Returns
        -------
        grad_wrt_wtil_G : 1D array, size G
            Entry g contains derivative of loss with respect to wtil_G[g]
        '''
        N = float(y_N.size)
        G = wtil_G.size
        C = float(self.C)

        grad_L2_wrt_wtil_G = np.zeros(G)  # TODO fixme.

        grad_sumBCE_wrt_wtil_G = np.zeros(G)  # TODO fixme.

        # Convert to base 2. Already done for you.
        if self.use_base2_for_BCE:
            grad_sumBCE_wrt_wtil_G = grad_sumBCE_wrt_wtil_G / np.log(2)

        # TODO create total gradient array.
        # Be sure to handle the rescaling by C and N correctly.
        grad_G = grad_L2_wrt_wtil_G + grad_sumBCE_wrt_wtil_G  # TODO fixme
        return grad_G

    # Helper methods

    def initialize_param_arr(self, xone_NG, y_N):
        ''' Initialize weight vectors according to this instance's recipe

        Args
        ----
        xone_NG : 2D array, size N x G (n_examples x n_features_including_bias)
            Input features, with last column of all ones
        y_N : 1D array, size N
            Binary labels for each example (either 0 or 1)

        Returns
        -------
        wtil_G : 1D array, size G (n_features_including_bias)
            Weight vector, where final entry is the bias
        '''
        F = self.num_features_excluding_bias
        G = F + 1
        if self.init_recipe == 'uniform_-1_to_1':
            wtil_G = self.random_state.uniform(-1, 1, size=G)
        elif self.init_recipe == 'uniform_-6_to_6':
            wtil_G = self.random_state.uniform(-6, 6, size=G)
        elif self.init_recipe == 'zeros':
            wtil_G = np.zeros(G)
        else:
            raise ValueError("Unrecognized init_recipe: %s" % self.init_recipe)
        return wtil_G

    def check_convergence(self, trace_steps, trace_loss,
                          trace_grad_L1_norm, trace_w):
        ''' Assess if current gradient descent run has converged

        We assume that at least 100 iters are needed to verify convergence.
        This might be abundantly cautious, but we'd rather be sure.

        Convergence is assessed on three criteria:
        * loss has stopped changing meaningfully over last 100 iters
            Measured by difference of loss from recent iters 100-50 to 50-0.
            Compared against the threshold attribute 'loss_converge_thr'
        * gradient is close enough to zero vector
            Measured by the L1 norm of the gradient vector at latest iteration.
            Compared against the threshold attribute 'grad_norm_converge_thr'
        * weights have not changed significantly over last 100 iters
            Compared against the threshold attribute 'param_converge_thr'

        If all 3 criteria are satisified, we return True.
        Otherwise, we return False.

        Args
        ----
        trace_steps : list of int
            Each entry is an iteration number
            Counts up from 0, 1, 2, ...
        trace_loss : list of scalars
            Each entry is the value of the loss at an iteration.
            Should be generally going down
        trace_grad_L1_norm : list of scalars
            Each entry is gradient's L1 norm at an iteration.
            Should be generally going down and approaching zero.
        trace_w : list of 1D arrays

        Returns
        -------
        did_converge : bool
            Boolean flag that indicates if the run has converged.
        '''
        iter_id = trace_steps[-1]
        # Assess convergence
        if iter_id < 100:
            is_loss_converged = False
            is_grad_converged = False
            is_param_converged = False
        else:
            # Criteria 1/3: has the loss stopped changing?
            # Calc average loss from 100-50 steps ago
            old_avg_loss = np.mean(trace_loss[-100:-50])
            # Calc average loss from 50-0 steps ago
            new_avg_loss = np.mean(trace_loss[-50:])
            loss_diff = np.abs(old_avg_loss - new_avg_loss)
            is_loss_converged = loss_diff < self.loss_converge_thr

            # Criteria 2/3: is the gradient close to zero?
            # Check if gradient is small enough
            is_grad_converged = (trace_grad_L1_norm[-1] <
                                 self.grad_norm_converge_thr)

            # Criteria 3/3: have weight vector parameters stopped changing?
            # Check if max L1 diff across all weight values is small enough
            max_param_diff = np.max(np.abs(trace_w[-100] - trace_w[-1]))
            is_param_converged = max_param_diff < self.param_converge_thr

        did_converge = (is_param_converged
                        and is_loss_converged and is_grad_converged)
        return did_converge

    def raise_error_if_diverging(
            self, trace_steps, trace_loss, trace_grad_L1_norm):
        ''' Raise error if current gradient descent run is diverging

        Will assess current trace and raise ValueError only if diverging.

        Divergence occurs when:
        * loss is going UP consistently over 10 iters, when should go DOWN.
        * loss is NaN or infinite
        * any entry of the gradient is NaN or infinite

        Divergence happens in gradient descent when step_size is set too large.
        If divergence is detected, we recommend using a smaller step_size.

        Args
        ----
        trace_steps : list of trace step numbers
            Counts up from 0, 1, 2, ...
        trace_loss : list of loss values
            Should be generally going down
        trace_grad_L1_norm : list of floats representing each grad's L1 norm
            Should be generally going down

        Returns
        -------
        Nothing

        Post Condition
        --------------
        Internal attribute `did_diverge` is set to True or False, as needed. 

        Raises
        ------
        ValueError if divergence is detected.
        '''
        n_completed_iters = len(trace_loss)
        loss = trace_loss[-1]
        L1_norm_grad = trace_grad_L1_norm[-1]
        did_diverge = False
        if np.isnan(loss):
            did_diverge = True
            reason_str = 'Loss should never be NaN'
        elif not np.isfinite(loss):
            did_diverge = True
            reason_str = 'Loss should never be infinite'
        elif np.isnan(L1_norm_grad):
            did_diverge = True
            reason_str = 'Grad should never be NaN'
        elif not np.isfinite(L1_norm_grad):
            did_diverge = True
            reason_str = 'Grad should never be infinite'

        # We need at least 6 completed steps to verify diverging...
        elif n_completed_iters >= 6:
            # Let's look at the 6 most recent steps we took, and compare:
            # * the average loss on steps 6-3
            # * the average loss on steps 3-0
            old_loss = np.median(trace_loss[-6:-3])
            new_loss = np.median(trace_loss[-3:])
            denom = (1e-10 + np.abs(old_loss))
            perc_change_last6steps = (new_loss - old_loss) / denom

            min_before = np.min(trace_loss[:-5])
            min_lastfew = np.min(trace_loss[-5:])
            noprogress_lastfew = min_lastfew >= min_before * 1.0001

            if perc_change_last6steps > 0.05 or noprogress_lastfew:
                did_diverge = True
                reason_str = 'Loss is increasing but should be decreasing!'

        self.did_diverge = did_diverge
        if did_diverge:
            hint_str = "Try a smaller step_size than current value %.3e" % (
                self.step_size)
            print("ALERT! Divergence detected. %s" % reason_str)
            print("Recent history of loss values:")
            M = np.minimum(10, n_completed_iters)
            for ii in range(M):
                print("iter %4d  loss % 11.6f" % (
                    trace_steps[-M+ii], trace_loss[-M+ii]))
            self.solver_status = "Divergence detected. %s. %s." % (
                reason_str, hint_str)
            raise ValueError(self.solver_status)


if __name__ == '__main__':
    # Toy problem
    # Logistic regression should be able to perfectly predict all 10 examples
    N = 10
    xoffset = 1.23
    x_NF = xoffset + np.hstack([
        np.linspace(-2, -1, 5), np.linspace(1, 2, 5)])[:, np.newaxis]
    y_N = np.hstack([np.zeros(5), np.ones(5)])

    lr = LogisticRegressionGradientDescent(
        C=1.0, init_recipe='zeros')

    # Prepare features by inserting column of all 1
    xone_NG = insert_final_col_of_all_ones(x_NF)

    print("Checking loss and grad when wtil_G is all zeros")
    wtil_G = np.zeros(2)
    print("wtil_G = %s" % str(wtil_G))
    print("loss(wtil_G) = %.3f" % lr.calc_loss(wtil_G, xone_NG, y_N))
    print("grad(wtil_G) = %s" % str(lr.calc_grad(wtil_G, xone_NG, y_N)))
