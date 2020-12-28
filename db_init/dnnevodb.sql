DROP TABLE IF EXISTS public.jobs;

CREATE TABLE public.jobs (
	id serial NOT NULL,
	submission_delay int4 NULL,
	mi numeric NULL,
	number_of_cores int4 NULL,
	cpu_time_spent_s numeric NULL,
	mips_per_core numeric NULL,
	wallclock_time_spent_s numeric NULL,
	CONSTRAINT jobs_pkey PRIMARY KEY (id)
);

DROP TABLE IF EXISTS public.experiments;

CREATE TABLE public.experiments (
	id serial NOT NULL,
	"name" varchar NULL,
	configuration_json varchar NULL,
	CONSTRAINT experiments_pkey PRIMARY KEY (id)
);

DROP TABLE IF EXISTS public.populations;

CREATE TABLE public.populations (
	id serial NOT NULL,
	"name" varchar NULL,
	experiment_id int4 NULL,
	CONSTRAINT populations_pkey PRIMARY KEY (id),
	CONSTRAINT populations_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

DROP TABLE IF EXISTS public.iterations;

CREATE TABLE public.iterations (
	id serial NOT NULL,
	iteration_no int4 NULL,
	population_id int4 NULL,
	CONSTRAINT iterations_pkey PRIMARY KEY (id),
	CONSTRAINT iterations_population_id_fkey FOREIGN KEY (population_id) REFERENCES populations(id)
);

DROP TABLE IF EXISTS public.individuals;

CREATE TABLE public.individuals (
	id serial NOT NULL,
	iteration_id int4 NULL,
	fitness numeric NULL,
	genotype varchar NULL,
	evaluation_time numeric NULL,
	parents varchar NULL,
	genealogy_index int4 NULL,
	CONSTRAINT individuals_pkey PRIMARY KEY (id),
	CONSTRAINT individuals_iteration_id_fkey FOREIGN KEY (iteration_id) REFERENCES iterations(id)
);
