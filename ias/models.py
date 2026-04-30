import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, DateTime, Numeric, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData

# Carga credenciales desde el .env del backend si existe
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', 'backend', '.env'))

def env(name, default=None):
  val = os.getenv(name)
  return val if val not in (None, '') else default

pg_user = env('PG_USER', 'autos_user')
pg_pass = env('PG_PASSWORD', '')
pg_host = env('PG_HOST', 'localhost')
pg_port = env('PG_PORT', '5432')
pg_db = env('PG_DATABASE', 'autoscastellon')

db_url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

Base = declarative_base(metadata=MetaData(schema='ac'))  # Esquema 'ac'

# Solo tabla relevante para regresión: Sale (ventas)
class Sale(Base):
  __tablename__ = 'Sale'
  SaleId = Column(UUID, primary_key=True)
  VehicleId = Column(UUID)
  Fecha = Column(DateTime)
  Precio = Column(Numeric(18, 2))

# Motor y sesión
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()
