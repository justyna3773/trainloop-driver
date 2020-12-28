DROP TABLE IF EXISTS public.metrics_table;

CREATE TABLE public.metrics_table (
	metrics_table_id bigserial NOT NULL,
	event_source varchar NOT NULL,
	source_object_id varchar NOT NULL,
	metric varchar NOT NULL,
	value varchar NOT NULL,
	inserted int8 NOT NULL,
	CONSTRAINT metrics_table_pkey PRIMARY KEY (metrics_table_id)
);
