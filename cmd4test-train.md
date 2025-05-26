validation Market
python test.py --config_file configs/Market/kat_market2.yml --checkpoint /data_sata/ReID_Group/ReID_Group/KANTransfarmers/logs/market_kat_transreid_20/transformer_60.pth --gpu_id 0

python test.py --config_file configs/Market/kat_market2.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT '/data_sata/ReID_Group/ReID_Group/KANTransfarmers/logs/market_kat_transreid_20/transformer_60.pth'

validation MSMT
python processor/validation.py --config_file configs/MSMT17/KAT_base_MSMT.yml --checkpoint /data_sata/ReID_Group/ReID_Group/KANTransfarmers/logs/msmt17_KAT_base/transformer_120.pkl --gpu_id 4

validation Occ_Duke
python processor/validation.py --config_file configs/OCC_Duke/KAT_base1.yml --checkpoint /data_sata/ReID_Group/ReID_Group/KANTransfarmers/logs/occ_duke_KAT_120/transformer_60.pkl --gpu_id 0

validation VeRi
python processor/validation.py --config_file configs/VeRi/KAN_base_veri.yml --checkpoint /data_sata/ReID_Group/ReID_Group/KANTransfarmers/logs/veri_KAT_base/transformer_120.pkl --gpu_id 2

validation Occuluded ReID
python processor/validation.py --config_file configs/occluded_reid/occluded_reid_KAT.yml --checkpoint /data_sata/ReID_Group/ReID_Group/KANTransfarmers/logs/occ_duke_KAT_base/transformer_120.pkl --gpu_id 2

#Training Market
python train.py --config_file configs/Market/kat_market2.yml MODEL.DEVICE_ID "('2')"

#Training MSMT
python train.py --config_file configs/MSMT17/KAT_base_MSMT.yml MODEL.DEVICE_ID "('4')"

#Training Occ_Duke
python train.py --config_file configs/OCC_Duke/KAT_base1.yml MODEL.DEVICE_ID "('5')"

#Training Veri
python train.py --config_file configs/VeRi/KAN_base_veri.yml MODEL.DEVICE_ID "('4')"


#Training Occluded_reid
python train.py --config_file configs/occluded_reid/occluded_reid_KAT.yml MODEL.DEVICE_ID "('0')"
