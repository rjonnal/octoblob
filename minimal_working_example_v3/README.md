### Minimum working example for ORG processing

1. Required data files (`HH-MM-SS.unp` designates any `.unp` file as named by the OCT acquisition software, in `HH-MM-SS.unp` format):

  * HH-MM-SS.unp
  * HH-MM-SS.xml
  
  Download [UNP1](https://www.dropbox.com/scl/fi/uxswhxf0jr1ywghh2czi8/16_53_25.unp?rlkey=nnm90nam0cfkvrufajvrcizoc&dl=0), [UNP2](https://www.dropbox.com/scl/fi/g03gaih40zwk5c4f8suef/16_58_12.unp?rlkey=o0blrmc41e886isi91fbnbitl&dl=0), [XML1](https://www.dropbox.com/scl/fi/867myyt46qw2j55ps9576/16_53_25.xml?rlkey=cwvymqk1sjx5o4eocmvnvr50f&dl=0), [XML2](https://www.dropbox.com/scl/fi/f023hnzf90hnmb9q04pun/16_58_12.xml?rlkey=xn19cvas98fw7n1ctyts12kbj&dl=0).
  
2. Required Python scripts:

  * `mwe_step_01_manual_dispersion_compensation.py`
  * `mwe_step_02_make_bscans.py`
  * `mwe_step_03_compute_org.py`
  * `functions.py`
  * `config.py`
  
3. To run, for example:

```python
python mwe_step_01_manual_dispersion_compensation.py 16_24_48.unp
python mwe_step_02_make_bscans.py 16_24_48.unp 
python mwe_step_03_compute_org.py 16_24_48_bscans
```

4. ORG results are stored in `HH-MM-SS/bscans/org`.
