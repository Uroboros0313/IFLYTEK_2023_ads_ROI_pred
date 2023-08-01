import os
import pathlib

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(__file__)))
LABEL = 'roi'
SEED = 1008

LAG_COLS = ['cum_income_cp_roi_norm', 'cum_income_cp_roi', 'income_cp_roi', 'spend', 'income']
TG_ENC_COLS = ['campaign_id', 'countries', 'product_id', 'account_id', 'post_type']
USELESS_COLS = ['uuid', 
                'ad_id', 
                'ad_set_id',
                'campaign_id', 
                'account_id', 
                'product_id',
                'datetime', 
                'date', 
                'countries', 
                'post_id_emb', 
                'post_type']
CUM_PREFIX_COLS = ['cum_spend',
                   'cum_impressions',
                   'cum_reach',
                   'cum_clicks',
                   'cum_engagement_nums',
                   'cum_post_shares',
                   'cum_post_reactions',
                   'cum_post_comments',
                   'cum_post_saves',
                   'cum_watch15s',
                   'cum_watch30s',
                   'cum_watch_p25',
                   'cum_watch_p50',
                   'cum_watch_p75',
                   'cum_watch_p95',
                   'cum_watch_p100',
                   'cum_watches',
                   'cum_bounces',
                   'cum_sessions',
                   'cum_session_duration',
                   'cum_add_cart_num',
                   'cum_add_payment_info',
                   'cum_initiates_checkout',
                   'cum_purchase',
                   'cum_income_1',
                   'cum_income_2']
