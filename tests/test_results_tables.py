import numpy as np
import pandas as pd
import pytest
from eog_clean.results_tables import correlation_rollup,CC


def test_correlation_rollup_keeps_absolute_values_and_equal_record_weights():
    rows=[]
    for record,starts in [(1,[0,10]),(2,[0])]:
        for start in starts:
            for channel,sign in [('A',1),('B',-1)]:
                magnitude=.8 if record==1 else .2
                row={'record':record,'start':start,'channel':channel,'method':'test'}
                for name in CC:row[name]=magnitude if '_abs_' in name else sign*magnitude
                rows.append(row)
    result=correlation_rollup(pd.DataFrame(rows),['method']).iloc[0]
    assert result.heog_abs_cc_after==pytest.approx(.5)
    assert result.heog_cc_after==pytest.approx(0)


def test_correlation_rollup_rejects_missing_record_identity():
    row={'record':np.nan,'start':0,'channel':'A','method':'test',**dict.fromkeys(CC,0.)}
    with pytest.raises(ValueError,match='identity'):
        correlation_rollup(pd.DataFrame([row]),['method'])
