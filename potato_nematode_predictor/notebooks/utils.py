import itertools
import multiprocessing
import fiona
import geopandas
from tqdm.autonotebook import tqdm
from rasterstats import zonal_stats, gen_zonal_stats


class RasterstatsMultiProc(object):
    """
    Inspired by https://github.com/perrygeo/python-rasterstats/blob/master/examples/multiproc.py
    """
    def __init__(self, tif, df=None, shp=None, stats=['min', 'max', 'median', 'mean', 'std'], all_touched=False):
            self.all_touched = all_touched
            self.df = df
            self.stats = stats
            self.tif = tif
            
    def calc_zonal_stats(self, band=1, prog_bar=True):
        gen = gen_zonal_stats(self.df, self.tif, stats=self.stats, band=band, geojson_out=True)
        length = self.df.shape[0]
        results = []
        
        if prog_bar:
            for result in tqdm(gen, total=length):
                results.append(result)
        else:
            for result in gen:
                results.append(result)
            
        results_df = geopandas.GeoDataFrame.from_features(results)
        results_df.crs = self.df.crs
        
        # Move the 'geometry' column to be the last column (https://stackoverflow.com/a/56479671/12045808)
        results_df = results_df[ [ col for col in results_df.columns if col != 'geometry' ] + ['geometry'] ]
        
        return results_df

    def calc_zonal_stats_multiproc(self):
        print("Be patient - it can easily take 30 min to calculate zonal stats.\n")
        with fiona.open(self.shp) as src:
            features = list(src)
            crs = src.crs

        # Create a process pool using all cores
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(cores) as pool:
            # parallel map
            results_lists = pool.map(self.zonal_stats_partial, self.chunks(features, cores))

        # flatten to a single list
        results = list(itertools.chain(*results_lists))
        assert len(results) == len(features)
        
        # Create geodataframe from the results
        results_df = geopandas.GeoDataFrame.from_features(results, crs=crs)
        
        # Move the 'geometry' column to be the last column (https://stackoverflow.com/a/56479671/12045808)
        results_df = results_df[ [ col for col in results_df.columns if col != 'geometry' ] + ['geometry'] ]
        
        return results_df 
    
    @staticmethod
    def chunks(data, n):
        """Yield successive n-sized chunks from a slice-able iterable."""
        for i in range(0, len(data), n):
            yield data[i:i+n]

    def zonal_stats_partial(self, feats):
        """Wrapper for zonal stats, takes a list of features"""
        return zonal_stats(feats, self.tif, all_touched=self.all_touched, stats=self.stats, geojson_out=True)
