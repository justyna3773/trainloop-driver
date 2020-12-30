from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from driver import logger

import os


MONITORING_DB_USERNAME = os.getenv('MONITORING_DB_USERNAME', 'samm')
MONITORING_DB_PASSWORD = os.getenv('MONITORING_DB_PASSWORD', 'samm')
MONITORING_DB_PORT = os.getenv('MONITORING_DB_PORT', '5432')
MONITORING_DB_HOST = os.getenv('MONITORING_DB_HOST', 'sammdb')
MONITORING_DB_NAME = os.getenv('MONITORING_DB_NAME', 'samm')

DNNEVO_DB_USERNAME = os.getenv('DNNEVO_DB_USERNAME', 'dnnevo')
DNNEVO_DB_PASSWORD = os.getenv('DNNEVO_DB_PASSWORD', 'dnnevo')
DNNEVO_DB_PORT = os.getenv('DNNEVO_DB_PORT', '5432')
DNNEVO_DB_HOST = os.getenv('DNNEVO_DB_HOST', 'dnnevodb')
DNNEVO_DB_NAME = os.getenv('DNNEVO_DB_NAME', 'dnnevo')


Base = declarative_base()
_session_jobs = None
_session_monitoring = None


def get_jobs_db_uri():
    uri = (f'postgresql+psycopg2://'
           f'{DNNEVO_DB_USERNAME}:{DNNEVO_DB_PASSWORD}'
           f'@{DNNEVO_DB_HOST}:{DNNEVO_DB_PORT}'
           f'/{DNNEVO_DB_NAME}'
           )

    return uri


def get_monitoring_db_uri():
    uri = (f'postgresql+psycopg2://'
           f'{MONITORING_DB_USERNAME}:{MONITORING_DB_PASSWORD}'
           f'@{MONITORING_DB_HOST}:{MONITORING_DB_PORT}'
           f'/{MONITORING_DB_NAME}'
           )

    return uri


class Job(Base):

    __tablename__ = 'jobs'

    job_id = Column('id', Integer, primary_key=True)
    submission_delay = Column('submission_delay', Integer)
    mi = Column('mi', Numeric)
    number_of_cores = Column('number_of_cores', Integer)
    cpu_time_spent_s = Column('cpu_time_spent_s', Numeric)
    mips_per_core = Column('mips_per_core', Numeric)
    wallclock_time_spent_s = Column('wallclock_time_spent_s', Numeric)

    def as_cloudlet_descriptor_dict(self):
        return {
            'jobId': self.job_id,
            'submissionDelay': int(self.submission_delay),
            'mi': int(self.mi),
            'numberOfCores': int(self.number_of_cores),
        }

    def __repr__(self):
        return (
            f'<Job('
            f'job_id={self.job_id},'
            f'submission_delay={self.submission_delay},'
            f'mi={self.mi},'
            f'number_of_cores={self.number_of_cores},'
            f'cpu_time_spent_s={self.cpu_time_spent_s},'
            f'mips_per_core={self.mips_per_core},'
            f'wallclock_time_spent_s={self.wallclock_time_spent_s}>'
        )


class MetricValue(Base):

    __tablename__ = 'metrics_table'

    metrics_table_id = Column('metrics_table_id', Integer, primary_key=True)
    event_source = Column("event_source", String)
    source_object_id = Column("source_object_id", String)
    metric = Column("metric", String)
    value = Column("value", String)
    inserted = Column("inserted", Integer)

    def as_json(self):
        return {
            'metrics_table_id': self.metrics_table_id,
            'event_source': self.event_source,
            'source_object_id': self.source_object_id,
            'metric': self.metric,
            'value': self.value,
            'inserted': self.inserted,
        }

    def __repr__(self):
        return (
            f'<MetricValue('
            f'metrics_table_id={self.metrics_table_id},'
            f'event_source={self.event_source},'
            f'source_object_id={self.source_object_id},'
            f'metric={self.metric},'
            f'value={self.value},'
            f'inserted={self.inserted}>'
        )


def _create_session_for_uri(uri):
    engine = create_engine(uri)
    Session = sessionmaker(bind=engine)
    return Session()


def init_jobs_db():
    global _session_jobs

    if _session_jobs is None:
        logger.info('Initializing jobs db')
        uri = get_jobs_db_uri()
        _session_jobs = _create_session_for_uri(uri)
        logger.info('Initialized jobs db')


def init_monitoring_db():
    global _session_monitoring

    if _session_monitoring is None:
        logger.info('Initializing monitoring db')
        uri = get_monitoring_db_uri()
        _session_monitoring = _create_session_for_uri(uri)
        logger.info('Initialized monitoring db')


def init_dbs():
    init_jobs_db()
    init_monitoring_db()


def get_jobs_between(start, end):
    from_db = _session_jobs.query(Job).filter(
        Job.submission_delay >= start
    ).filter(
        Job.submission_delay <= end
    ).order_by(Job.submission_delay.desc())

    jobs = [
        job.as_cloudlet_descriptor_dict()
        for job in from_db
    ]

    return jobs


def get_jobs_since(timestamp):
    return get_jobs_between(timestamp, time.time())


def get_jobs_from_last_s(timespan):
    now = time.time()
    return get_jobs_between(now - timespan, now)


def _get_metric_data_after(metric, timestamp, max_entries=None):
    query_result = _session_monitoring.query(MetricValue).\
        filter(MetricValue.inserted >= timestamp).\
        filter(MetricValue.metric == metric).\
        order_by(MetricValue.inserted.asc())

    if max_entries:
        query_result = query_result.limit(max_entries)

    dbos = []
    for dbo in query_result:
        dbos.append(dbo)

    return dbos


def _get_metric_values_after(metric, timestamp, max_entries=None):
    dbos = _get_metric_data_after(metric, timestamp, max_entries=max_entries)

    return [
        None if dbo.value is None else float(dbo.value)
        for dbo
        in dbos
    ]


def _get_metric_value_at(metric, timestamp):
    data = _get_metric_data_after(metric, timestamp, max_entries=1)

    if len(data) > 0:
        metric_value = float(data[0].value)
    else:
        metric_value = None

    return metric_value


def get_cores_count_after(timestamp):
    s_cores = _get_metric_values_after('coresUsedCountS', timestamp)
    m_cores = _get_metric_values_after('coresUsedCountM', timestamp)
    l_cores = _get_metric_values_after('coresUsedCountL', timestamp)

    return {
        's_cores': s_cores,
        'm_cores': m_cores,
        'l_cores': l_cores,
    }


def get_cores_count_at(timestamp):
    s_cores = _get_metric_value_at('coresUsedCountS', timestamp)
    m_cores = _get_metric_value_at('coresUsedCountM', timestamp)
    l_cores = _get_metric_value_at('coresUsedCountL', timestamp)

    return {
        's_cores': s_cores,
        'm_cores': m_cores,
        'l_cores': l_cores,
    }

if __name__ == '__main__':
    init_dbs()

    now = 1593347182
    jobs = get_jobs_since(now - 300)
    for j in jobs:
        print(j)

    print(get_cores_count_at(now*1000))
