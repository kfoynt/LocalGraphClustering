A=readSMAT('../../graph/minnesota.smat');
R=readSeed('../../graph/minnesota_R.smat');
[actual_length,ret_set]=MQI(A,R);
actual_length
ret_set