import json


class PlagueConfig:
    def __init__(
        self,
        target_model,
        attacker_model,
        evaluator_model,
        summariser_model,
        embedding_model,
        project,
        max_rounds,
        use_strategy_library=True,
        dataset="harmbench",
        verbose=False,
        max_workers=6,
        max_tries=1,
        primer_steps=2,
        finisher_type="normal",
        use_iterative_prompting: bool = True,
        use_planner: bool = True,
        max_search_results: int = 2,
        use_actor_plan: bool = False,
        *,
        attacker_port: int = 8201,
        evaluator_port: int = 8001,
        summariser_port: int = 7000,
        embedder_port: int = 7005,
        target_max_tokens: int = 8192,
        attacker_max_tokens: int = 4096,
        evaluator_max_tokens: int = 4096,
        summariser_max_tokens: int = 1024,
        llm_seed: int | None = None,
        use_vdb_judge: bool = False,
        sample_meta: dict | None = None,
    ):
        self.target_model = target_model
        self.attacker_model = attacker_model
        self.evaluator_model = evaluator_model
        self.summariser_model = summariser_model
        self.embedding_model = embedding_model
        self.project = project
        self.max_rounds = max_rounds
        self.dataset = dataset
        self.verbose = verbose
        self.max_workers = max_workers
        self.max_tries = max_tries
        self.primer_steps = primer_steps
        self.use_strategy_library = use_strategy_library
        self.finisher_type = finisher_type
        self.use_iterative_prompting = use_iterative_prompting
        self.use_planner = use_planner
        self.max_search_results = max_search_results
        self.use_actor_plan = use_actor_plan
        self.attacker_port = attacker_port
        self.evaluator_port = evaluator_port
        self.summariser_port = summariser_port
        self.embedder_port = embedder_port
        self.target_max_tokens = target_max_tokens
        self.attacker_max_tokens = attacker_max_tokens
        self.evaluator_max_tokens = evaluator_max_tokens
        self.summariser_max_tokens = summariser_max_tokens
        self.llm_seed = llm_seed
        self.use_vdb_judge = use_vdb_judge
        self.sample_meta = sample_meta or {}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
