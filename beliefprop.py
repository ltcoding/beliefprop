from graphframes.lib import AggregateMessages as AM
from pyspark.sql import functions as F
from functools import reduce
import operator
import math

#
# Uncomment lines below to set checkpoint directory for spark context
#
# CHECKPOINT_DIR = 'mydir'
# spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)


def log_add(lgx, lgy):
    """Gaussian logarithm for addition in log number system"""
    arg = lgy - lgx

    return lgx + F.when(arg > 30, arg) \
        .otherwise(F.log1p(F.exp(arg)))


def log_sub(lgx, lgy):
    """Gaussian logarithm for subtraction in log number system"""
    return F.log(F.abs(1 - F.exp(lgy - lgx)))


def log_mult(lgx, lgy):
    """Multiply in log number system"""
    return lgx + lgy


def log_div(lgx, lgy):
    """Divide in log number system"""
    return lgx - lgy


class MarkovRandomField(object):
    """Pyspark Graphframes implementation of Belief Propagation for Markov Random Fields where
        the support of each random variable is the same

        Parameters
        ----------
        phi : str
            Name of column of prior probabilities
        psi : List[List[float]]
            psi[i][j] represents edge potential for edge e_ij
        momentum : float
            Also known as damping factor

        Methods
        -------
        run_bp(gx, num_iter, is_checkpoint=False):
            Runs belief propagation given a GraphFrame

        References
        ----------
        J. S. Yedidia, W. T. Freeman and Y Weiss, "Understanding Belief Propagation and its Generalizations,"
        in Exploring Artificial Intelligence in the New Millennium, Morgan Kaufmann, 2003, pp. 239-269

        http://people.csail.mit.edu/billf/publications/Understanding_Belief_Propogation.pdf
        """

    def __init__(self, phi, psi, momentum=0.5):

        self.phi = phi
        self.psi = psi

        self.momentum = momentum

        self.n = len(psi)
        self.n_src = len(self.psi)  # Support size of random variables in src column of edge table
        self.n_dst = len(self.psi[0])  # Support size of random variables in dst column of edge table

        self.phi_cols = [self._phi(k=k) for k in range(self.n)]

        # Column names for prev messages in edge dataframe
        self.mij_colnames = [self._mij(k) for k in range(self.n_dst)]
        self.mji_colnames = [self._mji(k) for k in range(self.n_src)]

        # Column names for newly calculated messages in edge dataframe
        self.new_mij_colnames = [self._new_mij(k) for k in range(self.n_dst)]
        self.new_mji_colnames = [self._new_mji(k) for k in range(self.n_src)]

        self.oldm2newm = {self.mij_colnames[k]: self.new_mij_colnames[k]
                          for k in range(self.n_dst)}

        self.oldm2newm.update({
            self.mji_colnames[k]: self.new_mji_colnames[k] for k in range(self.n_src)
        })

        self.msg_colnames = self.mij_colnames + self.mji_colnames
        self.new_msg_colnames = self.new_mij_colnames + self.new_mji_colnames

        # TODO: Generalize if nodes have different supports
        self.belief_cols = [self._belief(k) for k in range(self.n)]

        self.agg_cols = [self._aggMess(k) for k in range(self.n_dst)]

    @staticmethod
    def _belief(k):
        """Return name of column representing belief of vertex in event k"""
        return 'belief_{k}'.format(k=k)

    @staticmethod
    def _mij(k):
        """Return name of column in edge table representing message for belief in event k sent from node i->j """
        return 'mij_{k}'.format(k=k)

    @staticmethod
    def _mji(k):
        """Return name of column in edge table representing message for belief in event k sent from node j->i """
        return 'mji_{k}'.format(k=k)

    @classmethod
    def _new_mij(cls, k):
        """Return column name in edge table of newly calculated message for belief in event k sent from node i->j """
        return 'n_{}'.format(cls._mij(k))

    @classmethod
    def _new_mji(cls, k):
        """Return column name in edge table of newly calculated message for belief in event k sent from node j->i"""
        return 'n_{}'.format(cls._mji(k))

    @staticmethod
    def _aggMess(k=None, node=None):
        """Return column name for aggregate messages for belief in event k for node (node can be 'src' or 'dst')"""
        return '_'.join([str(x) for x in ['AM', k, node] if x is not None])

    def _phi(self, k=None, node=None):
        """Return column name for phi (prior belief) for event k of node (node can be 'src' or 'dst')"""
        return '_'.join([str(x) for x in [self.phi, k, node] if x is not None])

    @staticmethod
    def _rename_cols(df, old2new_d):
        """Rename columns in df as specified in a dictionary mapping old to new column name"""
        return df.select(*[F.col(c).alias(old2new_d.get(c, c)) for c in df.columns])

    @staticmethod
    def _has_missing_v(e, v, ecol='src'):
        """Returns if there are any nodes in ECOL column of edge table that do not exist in vertex table"""
        return e.select(ecol).join(v, on=(v['id'] == e[ecol]), how='left_anti').rdd.isEmpty()

    def _join_v2e(self, edges, vertices, vcols, on_v='id', on_e='src'):
        """Add vertex properties to edge dataframe for each vertex in "src" (or "dst")

        Parameters
        ----------
        edges : DataFrame
            Edge dataframe (contains src and dst columns)
        vertices : DataFrame
            Vertex dataframe
        vcols : List[str]
            List of columns in vertex dataframe to keep after join
        on_v : str, default='id'
            Column of vertex dataframe to join on
        on_e : {'src', 'dst'}
            Column of edge dataframe to join on

        Returns
        -------
        DataFrame
            Edge dataframe where vertex properties have been added for src vertices (or dst vertices)
        """

        edges = edges.join(
            vertices.select(*vcols),
            on=(edges[on_e] == vertices[on_v]),
            how='left_outer'
        ).drop(on_v)

        edges = self._rename_cols(edges, {k: k + '_' + on_e for k in vcols})
        return edges

    def _overwrite_msgs(self, e, msg_colnames=None):
        """Overwrite old messages with new messages using damping, i.e., momentum (leave new message columns in place)

        Implements: m_{i+1} = lambda * m_{i} + (1 - lambda) * m_new_{i}

        Parameters
        ----------
        e : DataFrame
            Edge dataframe containing messages
        msg_colnames : List[str]
            List of names of message columns to update

        Returns
        -------
        DataFrame
            Edge dataframe with messages updated
        """

        msg_colnames = msg_colnames if msg_colnames is not None else self.msg_colnames
        for c in msg_colnames:
            e = e.withColumn(c, self.momentum * F.col(c) +
                             (1 - self.momentum) * F.col(self.oldm2newm[c]))

        return e

    @staticmethod
    def _normalize_log_cols(e, cols):
        """Numerically stable method to normalize columns with log number system (helper function)

        Parameters
        ----------
        e : DataFrame
            DataFrame containing columns to normalize (already in log number system)
        cols : List[str]
            List of columns to normalize

        Returns
        -------
        DataFrame
            Dataframe with columns normalized in log number system
        """

        total = 'tmp_log_total'
        tmp_max = 'tmp_max'

        e = e.withColumn(tmp_max, F.greatest(*[F.col(c) for c in cols]))

        # Subtract max for numerical stability
        e = e.withColumn(total, F.col('tmp_max') + \
                         F.log(reduce(operator.add,
                                      [F.exp(F.col(c) - F.col('tmp_max')) for c in cols])))

        for c in cols:
            e = e.withColumn(c, F.col(c) - F.col(total))

        return e.drop(*[total, tmp_max])

    def normalize_msgs(self, e):
        """Normalizes messages mij and mji

        Parameters
        ----------
        e : DataFrame
            Edge dataframe containing message columns

        Returns
        -------
        DataFrame
            Edge dataframe with message columns normalized
        """
        e = self._normalize_log_cols(e, self.mij_colnames)
        e = self._normalize_log_cols(e, self.mji_colnames)
        return e

    def _mult_incoming_msgs(self, gx, n=None):
        """Multiplies all incoming messages coming into node i (performs sum in log number system)

        Precondition: Messages should be in log number system

        Parameters
        ----------
        gx : GraphFrame
        n : int
            Support (number of events) of random variable nodes

        Returns
        -------
        DataFrame
            Vertex dataframe with columns for id, and product of messages for each event
        """

        n = self.n if n is None else n

        # Multiplying takes the form of addition in log number system
        aggregates = [gx.aggregateMessages(F.sum(AM.msg).alias(self._aggMess(k)),
                                           sendToSrc=AM.edge[self._mji(k)],
                                           sendToDst=AM.edge[self._mij(k)])
                      for k in range(n)]

        acc = aggregates[0]
        for x in aggregates[1:]:
            acc = acc.join(x, on='id', how='inner')
            acc = acc.cache()
        return acc

    def _get_residuals(self, e, v1_cols=None, v2_cols=None, added_colname='res'):
        """Calculate the residuals for two vectors (sum of square differences). (helper function)

        Parameters
        ----------
        e : DataFrame
            Edge dataframe containing message values from previous and current iteration
        v1_cols : List[str], optional
            List of columnames for messages in previous iteration
        v2_cols : List[str], optional
            List of columnames for messages in current iteration
        added_colname : str, optional
            Name of column to store residuals

        Returns
        -------
        DataFrame
            Edge dataframe with new column added containing residuals
        """
        v1_cols = self.mij_colnames if v1_cols is None else v1_cols
        v2_cols = self.new_mij_colnames if v2_cols is None else v2_cols

        n1 = len(v1_cols)
        assert n1 == len(v2_cols)

        # Sum of squared differences
        e = e.withColumn(added_colname,
                         F.sqrt(reduce(operator.add,
                                       [F.pow(F.col(v1_cols[i]) - F.col(v2_cols[i]), 2)
                                        for i in range(n1)])))

        return e

    def get_msg_residuals(self, e, rij='rij', rji='rji'):
        """Find residuals for messages mij and mji

        Parameters
        ----------
        e : DataFrame
            Edge dataframe
        rij : str, optional
            Column name to use for residual rij
        rji : str, optional
            Column name to use for residual rji

        Returns
        -------
        DataFrame
            Edge dataframe where residuals for messages mij and mji have been added in columns rij and rji respectively
        """
        e = self._get_residuals(e, self.mij_colnames, self.new_mij_colnames, added_colname=rij)
        e = self._get_residuals(e, self.mji_colnames, self.new_mji_colnames, added_colname=rji)

        return e

    def set_belief(self, gx):
        """Marginalize to get beliefs for each node.

        Precondition: Have already run previous iterations of BP so message columns in self.msg_colnames
          exist in the edge table

        Parameters
        ----------
        gx : GraphFrame
            Graph with vertices for which to we wish to find associated marginal probabilities (beliefs)

        Returns
        -------
        GraphFrame
            Resultant graph where list of columns specified by self.belief_cols has been
            added to the vertex dataframe giving the marginalized probabilities (beliefs)

        References
        ----------
        Eq. (13) of Yedidia et. al. mentioned in references of class docstring
        """

        prod_msgs = self._mult_incoming_msgs(gx, n=self.n_dst)  # aggregate incoming messages
        v = gx.vertices.join(prod_msgs, on='id')

        # See Eq. (13) of Yedidia et. al. cited in References
        for k in range(self.n_src):
            v = v.withColumn(self._belief(k),
                             F.col(self._phi(k)) + F.col(self._aggMess(k)))

        # Normalize probabilities
        v = self._normalize_log_cols(v, self.belief_cols)

        gx = graphframes.GraphFrame(v, gx.edges)
        return gx

    def update_msg_a2b(self, gx, a='src'):
        """Update messages passed

        Parameters
        ----------
        gx : GraphFrame
            GraphFrame to update message passing on
        a : {'src', 'dst'}
            Origin of message being passed. 'src' indicates i -> j and "dest" indicates j -> i

        Returns
        -------
        DataFrame
            Edge dataframe where messages columns specified in self.new_mij_colnames (resp self.new_mji_colnames)
            were updated if a == 'src' (resp a == 'dst')

        References
        ----------
        Eq. (14) of Yedidia et. al. mentioned in references of class docstring
        """

        prod_msgs = self._mult_incoming_msgs(gx, n=self.n_dst)  # aggregate incoming messages

        # Add product terms for source and destination nodes in edge table
        e = gx.edges
        e = self._join_v2e(e, prod_msgs, vcols=['id'] + self.agg_cols, on_e='src')
        e = self._join_v2e(e, prod_msgs, vcols=['id'] + self.agg_cols, on_e='dst')

        # TODO: generalize for when n_out != n_in
        # Update mij
        # See Eq. (14) of Yedidia et. al. cited in References
        if a == 'src':
            for k in range(self.n_dst):
                terms_ij = [e[self._phi(i, 'src')] + self.psi[i][k] + e[self._aggMess(i, 'src')] - e[self._mji(i)]
                            for i in range(self.n_src)]
                e = e.withColumn(self._new_mij(k), reduce(log_add, terms_ij))
            e = self._normalize_log_cols(e, self.new_mij_colnames)

        # Update mji
        else:
            for k in range(self.n_src):
                terms_ji = [e[self._phi(j, 'dst')] + self.psi[j][k] + e[self._aggMess(j, 'dst')] - e[self._mij(j)]
                            for j in range(self.n_dst)]
                e = e.withColumn(self._new_mji(k), reduce(log_add, terms_ji))
            e = self._normalize_log_cols(e, self.new_mji_colnames)

        drop_cols = [self._aggMess(i, 'src') for i in range(self.n_src)] + \
                    [self._aggMess(i, 'dst') for i in range(self.n_dst)]

        e = e.drop(*drop_cols)

        return e

    def run_bp(self, gx, num_iter, is_checkpoint=False):
        """Runs the belief propagation algorithm over many iterations

        Parameters
        ----------
        gx : GraphFrame
            Graphical model to run BP on
        num_iter : int
            Number of iterations to run algorithm
        is_checkpoint : bool, default=False
            Flag for whether edge dataframe should be checkpointed each iteration

        Returns
        -------
        GraphFrame
            Resultant graph where list of columns specified by self.belief_cols has been
            added to the vertex dataframe giving the marginalized probabilities (beliefs)
        """
        v = gx.vertices
        e = gx.edges

        # Ensure all vertices appearing in edge table also appear in vertex table
        assert not self._has_missing_v(e, v, 'src') == 0, 'edge\'s src node missing from vertex table'
        assert not self._has_missing_v(e, v, 'dst') == 0, 'edge\'s dst node missing from vertex table'

        # Add priors for source and destination nodes in edge table
        e = self._join_v2e(e, v, vcols=(['id'] + self.phi_cols), on_e='src')
        e = self._join_v2e(e, v, vcols=(['id'] + self.phi_cols), on_e='dst')

        # Initialize messages to log(1) = 0
        e = e.select('*', *[F.lit(0).alias(c) for c in self.msg_colnames])

        gx = graphframes.GraphFrame(v, e)

        for it in range(num_iter):

            # Update mji
            e = self.update_msg_a2b(gx, a='dst')
            e = self._overwrite_msgs(e, msg_colnames=self.mji_colnames)
            gx = graphframes.GraphFrame(v, e)

            # Update mij
            e = self.update_msg_a2b(gx, a='src')
            e = self._overwrite_msgs(e, msg_colnames=self.mij_colnames)

            # TODO: Get schedule for updating messages, e.g., with residuals
            # e = self.get_msg_residuals(e, 'rij', 'rji')

            if is_checkpoint:
                # Ensure spark.sparkContext.setCheckpointDir() has been set
                e = e.checkpoint()

            gx.unpersist(blocking=True)
            gx = graphframes.GraphFrame(v, e)

        gx = self.set_belief(gx)

        return gx


class SNARE(MarkovRandomField):
    """An implementation of the SNARE belief propagation algorithm in Pyspark GraphFrames

    References
    ----------
    M. McGlohon, S. Bay, M. G. Anderle, D. M. Steier and C. Faloutsos, "SNARE: A Link Analytic System for
    Graph Labeling and Risk Detection," in Proc. of the 15th ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining, Paris, France, June 28 - July 1, 2009, pp. 1265–1274

    http://www.cs.cmu.edu/~mmcgloho/pubs/snare.pdf
    """

    def __init__(self, eps=0.03, phi_col='prior', *args, **kwargs):
        """Initialize input probabilities to algorithm

        Parameters
        ----------
        eps : float
            Noise parameter for edge potentials in SNARE
        phi_col : str
            Name of column containing prior probabilities
        args
        kwargs
        """
        self.eps = eps

        # Use log number system
        log_eps = math.log(eps)
        log_1m_eps = math.log(1 - eps)

        psi = [[log_1m_eps, log_eps], [log_eps, log_1m_eps]]

        super().__init__(psi=psi, phi=phi_col, *args, **kwargs)
