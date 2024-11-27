import os
import sqlite3
import pandas as pd

class UniprotMappingsDBHandler:
    def __init__(self, db_path: str ,db_name: str = 'uniprot_mappings.db'):
        self.db = os.path.join(db_path, db_name)
        self.db_name = db_name
        
    def create_tables(self) -> None:
        self.create_mapping_table()
        self.disable_indices()
    
    def create_mapping_table(self) -> None:
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS mappings')
        cursor.execute('''
            CREATE TABLE mappings (
                UniProtKB_AC TEXT,
                UniProtKB_ID TEXT,
                GeneID TEXT,
                RefSeq TEXT,
                PDB TEXT,
                UniParc TEXT,
                NCBI_taxon TEXT,
                EMBL_CDS TEXT,
                Ensembl TEXT,
                PRIMARY KEY (UniProtKB_AC, UniProtKB_ID, GeneID, RefSeq, PDB, UniParc, NCBI_taxon, EMBL_CDS, Ensembl)
            )
        ''')
        conn.commit()
        conn.close()
        
    def disable_indices(self) -> None:
        conn = sqlite3.connect(self.db)
        # Disabling indices and other performance-related settings
        conn.execute('PRAGMA synchronous = OFF')
        conn.execute('PRAGMA journal_mode = OFF')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA cache_size = -1048576')  # 1 GB cache size (1,048,576 KB) 
        conn.commit()
        conn.close()
        
    def enable_indices(self) -> None:
        conn = sqlite3.connect(self.db)
        # Re-enabling indices and other settings
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA temp_store = DEFAULT')        
        conn.execute('PRAGMA cache_size = -2000')  # Reset to default cache size  
        conn.commit()
        conn.close()  
                        
    def reindex(self) -> None:
        self.enable_indices()        
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_UniProtKB_AC ON mappings (UniProtKB_AC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_UniProtKB_ID ON mappings (UniProtKB_ID)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_GeneID ON mappings (GeneID)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_RefSeq ON mappings (RefSeq)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_UniParc ON mappings (UniParc)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_EMBL_CDS ON mappings (EMBL_CDS)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_Ensembl ON mappings (Ensembl)')
        conn.commit()
        conn.close()       

    def reindex_column(self, field: str) -> None:
        self.enable_indices()        
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{field} ON mappings ({field})')
        conn.commit()
        conn.close()

    def batch_insert_mappings(self, mappings: list[tuple[str]]) -> None:
        self.disable_indices()
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.executemany('''
                           INSERT OR REPLACE INTO mappings (
                                UniProtKB_AC,
                                UniProtKB_ID,
                                GeneID,
                                RefSeq,
                                PDB,
                                UniParc,
                                NCBI_taxon,
                                EMBL_CDS,
                                Ensembl
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''' , mappings)
        conn.commit()
        conn.close()                
                
    def select(self, codes: list[str], field: str, return_fields: list[str] = None) -> list[tuple]:
        
        # which fields to return
        if return_fields is None:
            select_fields = '*'
        else:
            select_fields = ', '.join(return_fields)

        # set tabel
        table = 'mappings'

        # which records based on ncbi_codes
        # the query contains ?,?,?,?, with the cursor.execute(query, codes) the values are inserted
        records = f"({','.join(['?']*len(codes))})" # (?,?,?,?) for each entry in ncbi_code      
        query = 'SELECT {} FROM {} WHERE {} IN {}'.format(select_fields, table, field, records)
        
        # execute query
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute(query, codes)
        result = cursor.fetchall()  # fetchall() gets all, fetchone() just the next in the result list
        conn.close()
        
        return result
    
    def select_all(self, table: str = 'mappings') -> list[tuple]:
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table}')
        result = cursor.fetchall()  # fetchall() gets all, fetchone() just the next in the result list
        conn.close()
        
        return result
    
    def fetch_records_as_dataframe(self, codes: list[str], field: str, return_fields: list[str] = None) -> pd.DataFrame:

        table = 'mappings'

        # which fields to return
        if return_fields is None:
            select_fields = '*'
        else:
            select_fields = ', '.join(return_fields)

        # combien query
        records = f"({','.join(['?']*len(codes))})" # (?,?,?,?) for each entry in ncbi_code       
        query = 'SELECT {} FROM {} WHERE {} IN {}'.format(select_fields, table, field, records)

        conn = sqlite3.connect(self.db)
        df = pd.read_sql_query(query, conn, params = codes)
        conn.close()
        
        return df
    
    def get_columns(self) -> list[str]:
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute('PRAGMA table_info(mappings)')
        fields = cursor.fetchall()
        conn.close()

        # Extract column names from the fetched data
        column_names = [field[1] for field in fields]
        
        return column_names
