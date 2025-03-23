import requests
from tqdm import tqdm

base_url = "https://www.ncei.noaa.gov/data/nclimgrid-daily/archive/"

file_paths = [
    '2013/nclimgrid-daily_v1-0-0_complete_s20130101_e20130131_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130201_e20130228_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130301_e20130331_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130401_e20130430_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130501_e20130531_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130601_e20130630_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130701_e20130731_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130801_e20130831_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20130901_e20130930_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20131001_e20131031_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20131101_e20131130_c20220907.tar.gz',
    '2013/nclimgrid-daily_v1-0-0_complete_s20131201_e20131231_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140101_e20140131_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140201_e20140228_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140301_e20140331_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140401_e20140430_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140501_e20140531_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140601_e20140630_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140701_e20140731_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140801_e20140831_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20140901_e20140930_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20141001_e20141031_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20141101_e20141130_c20220907.tar.gz',
    '2014/nclimgrid-daily_v1-0-0_complete_s20141201_e20141231_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150101_e20150131_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150201_e20150228_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150301_e20150331_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150401_e20150430_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150501_e20150531_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150601_e20150630_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150701_e20150731_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150801_e20150831_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20150901_e20150930_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20151001_e20151031_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20151101_e20151130_c20220907.tar.gz',
    '2015/nclimgrid-daily_v1-0-0_complete_s20151201_e20151231_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160101_e20160131_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160201_e20160229_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160301_e20160331_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160401_e20160430_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160501_e20160531_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160601_e20160630_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160701_e20160731_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160801_e20160831_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20160901_e20160930_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20161001_e20161031_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20161101_e20161130_c20220907.tar.gz',
    '2016/nclimgrid-daily_v1-0-0_complete_s20161201_e20161231_c20220907.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170101_e20170131_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170201_e20170228_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170301_e20170331_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170401_e20170430_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170501_e20170531_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170601_e20170630_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170701_e20170731_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170801_e20170831_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20170901_e20170930_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20171001_e20171031_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20171101_e20171130_c20220908.tar.gz',
    '2017/nclimgrid-daily_v1-0-0_complete_s20171201_e20171231_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180101_e20180131_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180201_e20180228_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180301_e20180331_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180401_e20180430_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180501_e20180531_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180601_e20180630_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180701_e20180731_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180801_e20180831_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20180901_e20180930_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20181001_e20181031_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20181101_e20181130_c20220908.tar.gz',
    '2018/nclimgrid-daily_v1-0-0_complete_s20181201_e20181231_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190101_e20190131_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190201_e20190228_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190301_e20190331_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190401_e20190430_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190501_e20190531_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190601_e20190630_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190701_e20190731_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190801_e20190831_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20190901_e20190930_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20191001_e20191031_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20191101_e20191130_c20220908.tar.gz',
    '2019/nclimgrid-daily_v1-0-0_complete_s20191201_e20191231_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200101_e20200131_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200201_e20200229_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200301_e20200331_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200401_e20200430_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200501_e20200531_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200601_e20200630_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200701_e20200731_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200801_e20200831_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20200901_e20200930_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20201001_e20201031_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20201101_e20201130_c20220908.tar.gz',
    '2020/nclimgrid-daily_v1-0-0_complete_s20201201_e20201231_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210101_e20210131_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210201_e20210228_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210301_e20210331_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210401_e20210430_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210501_e20210531_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210601_e20210630_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210701_e20210731_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210801_e20210831_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20210901_e20210930_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20211001_e20211031_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20211101_e20211130_c20220908.tar.gz',
    '2021/nclimgrid-daily_v1-0-0_complete_s20211201_e20211231_c20220908.tar.gz',
]

def check_links(base_url, paths):
    valid_links = []
    invalid_links = []

    for path in tqdm(paths, desc="Checking links"):
        full_url = base_url + path
        try:
            response = requests.head(full_url, timeout=10)
            if response.status_code == 200:
                valid_links.append(full_url)
            else:
                invalid_links.append((full_url, response.status_code))
        except requests.RequestException as e:
            invalid_links.append((full_url, str(e)))

    return valid_links, invalid_links


valid_links, invalid_links = check_links(base_url, file_paths)

print("\nvalid links:")
for link in valid_links:
    print(link)

print("\ninvalid links:")
for link, reason in invalid_links:
    print(f"{link} - {reason}")
