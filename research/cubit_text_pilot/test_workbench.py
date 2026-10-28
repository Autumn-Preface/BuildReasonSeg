"""Behavioral gates, real authorized four-image provenance + isolated non-human drafts."""
import copy, json, sys, unittest, uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import BASE, REPO, Store, empty_edit
MANIFEST=REPO/'reports/cubit_text_pilot/20261029/first_four.json'
class DraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.store=Store(MANIFEST,BASE/'qa_drafts'/('unit-'+uuid.uuid4().hex))
    def edit(self,key='CR1_177'):
        e=empty_edit();e.update(operator='TOOL_TEST_NOT_HUMAN',instruction_text='工程测试占位文字，不属于人工标注。',target_instance_ids=[self.store.sample(key)['instances'][0]['instance_id']]);return e
    def test_01_four_real_roundtrips(self):
        for k in self.store.samples:
            e=self.edit(k);r=self.store.save(k,None,0,e);s=Store(MANIFEST,self.store.root);loaded=s.load(k,r['annotation_id']);self.assertEqual(loaded['edit'],e);self.assertEqual(loaded['source']['image_sha256'],s.sample(k)['image_sha256']);self.assertFalse(loaded['formal_export_enabled'])
    def test_02_revision_history_conflict(self):
        e=self.edit();r=self.store.save('CR1_177',None,0,e);before=self.store.revisions('CR1_177',r['annotation_id'])[0].read_bytes();e['notes']='工程第二版';r2=self.store.save('CR1_177',r['annotation_id'],1,e)
        self.assertEqual(len(r2['history']),2);self.assertIsNotNone(r2['history'][1]['previous_revision_sha256']);self.assertEqual(self.store.revisions('CR1_177',r['annotation_id'])[0].read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'版本冲突'):self.store.save('CR1_177',r['annotation_id'],1,e)
    def test_03_group_geometry_roundtrip(self):
        k='SP0_794';e=self.edit(k);e['target_instance_ids']=[i['instance_id'] for i in self.store.sample(k)['instances']];e['references']=[dict(id='ref-test',name='工具测试框，非人工参照物',bbox_normalized=[.1,.2,.3,.4])];e['regions']=[dict(kind='unknown',bbox_normalized=[.6,.6,.8,.8],note='工程测试')];e['relations']=[dict(reference_id='ref-test',target_instance_ids=e['target_instance_ids'],relation='测试关系',evidence='仅保存验证')];r=self.store.save(k,None,0,e);self.assertEqual(self.store.load(k,r['annotation_id'])['edit'],e)
    def test_04_no_human_confirmation(self):
        e=self.edit();e['review_status']='accepted'
        with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
    def test_05_foreign_target_rejected(self):
        e=self.edit();e['target_instance_ids']=[self.store.sample('CR1_40')['instances'][0]['instance_id']]
        with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
    def test_06_nonfinite_and_outside_geometry(self):
        for b in ([0,0,float('nan'),1],[0,0,2,1],[.5,0,.4,1]):
            e=self.edit();e['regions']=[dict(kind='unknown',bbox_normalized=b,note='')]
            with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
    def test_07_internal_input_leak_rejected(self):
        for t in ('CR1_177','cubit-v1/train/a','class 0','类别 1','L0001','direct_target'):
            e=self.edit();e['instruction_text']=t
            with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
    def test_08_source_write_path_rejected(self):
        for root in (self.store.source,REPO/'reports/cubit_text_prep/20261029',REPO/'delivery_src'):
            with self.assertRaises(ValueError):Store(MANIFEST,root)
    def test_09_traversal_id_rejected(self):
        with self.assertRaises(ValueError):self.store.load('CR1_177','../../raw')
        with self.assertRaises(ValueError):self.store.sample('SP0_1011')
    def test_10_unknown_fields_and_author(self):
        e=self.edit();e['operator']=''
        with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
        e=self.edit();e['formal_export_enabled']=True
        with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
    def test_11_orphan_relation(self):
        e=self.edit();e['relations']=[dict(reference_id='ref-no',target_instance_ids=e['target_instance_ids'],relation='above',evidence='')]
        with self.assertRaises(ValueError):self.store.save('CR1_177',None,0,e)
    def test_12_default_unknown_and_empty(self):
        e=empty_edit();self.assertEqual(e['valid_domain'],'unknown');self.assertEqual(e['target_instance_ids'],[]);self.assertEqual(e['instruction_text'],'')
if __name__=='__main__':unittest.main(verbosity=2)
