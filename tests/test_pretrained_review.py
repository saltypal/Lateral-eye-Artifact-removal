"""Prevent unavailable pretrained releases from becoming fabricated evaluations."""
import json
from pathlib import Path
import pytest
from eog_clean.pretrained_review import validate_blocked_candidates, verified_table_parent


def configuration():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/frontal_pretrained.json').read_text())


def test_reject_training_or_unsupported_eligible_release():
    review = configuration()
    review['training_allowed'] = True
    with pytest.raises(ValueError, match='pretrained-only'):
        validate_blocked_candidates(review)
    review = configuration()
    review['candidates'][0]['eligible'] = True
    with pytest.raises(ValueError, match='verified inference adapter'):
        validate_blocked_candidates(review)


def test_reject_invented_checkpoint_for_blocked_candidate():
    review = configuration()
    review['candidates'][0]['checkpoint_sha256'] = 'invented'
    with pytest.raises(ValueError, match='invented checkpoint'):
        validate_blocked_candidates(review)


def test_parent_payload_tampering_and_wrong_identity(tmp_path):
    payload = tmp_path/'table.csv'
    payload.write_text('altered')
    manifest = {'run_id':'fixture','git_sha':'source','complete':True,
                'files':[{'path':'table.csv','bytes':7,'sha256':'incorrect'}]}
    (tmp_path/'artifact_manifest.json').write_text(json.dumps(manifest))
    verify = verified_table_parent(tmp_path, 'fixture', 'source')
    with pytest.raises(ValueError, match='payload differs'):
        verify('table.csv')
    with pytest.raises(ValueError, match='source SHA differs'):
        verified_table_parent(tmp_path, 'fixture', 'wrong')
